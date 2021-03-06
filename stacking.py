# Modified from: Reid Johnson
# http://nbviewer.ipython.org/github/anfibil/cse40647.sp14/blob/master/32%20-%20Stacking%20%26%20Blending.ipynb
# which was modded from Kemal Eren (https://github.com/kemaleren/scikit-learn/blob/stacking/sklearn/ensemble/stacking.py)
#
# Generates a stacking/blending of base models. Cross-validation is used to 
# generate predictions from base (level-0) models that are used as input to a 
# combiner (level-1) model.
from sklearn.metrics import roc_curve, auc
import numpy as np
#from itertools import izip
#from sklearn.grid_search import IterGrid
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils.validation import assert_all_finite

# TODO: Built-in nested cross validation, re-using base classifiers,
#       to pick best stacking method.
# TODO: Access to best, vote, etc. after training.

__all__ = [
    "Stacking",
    "StackingFWL",
    'estimator_grid'
]

#def estimator_grid(*args):
#    """Generate candidate estimators from a list of parameter values 
#    on the combination of the various parameter lists given.
#
#    Parameters
#    ----------
#    args : array
#        List of classifiers and corresponding parameters.
#
#    Returns
#    -------
#    result : array
#        The generated estimators.
#    """
#    result = []
#    pairs = izip(args[::2], args[1::2])
#    for estimator, params in pairs:
#        if len(params) == 0:
#            result.append(estimator())
#        else:
#            for p in IterGrid(params):
#                result.append(estimator(**p))
#    return result


class MRLR(ClassifierMixin):
    """Converts a multi-class classification task into a set of
    indicator regression tasks.

    Ting, K.M., Witten, I.H.: Issues in stacked generalization.

    """
    def __init__(self, regressor, stackingc, **kwargs):
        self.estimator_ = regressor
        self.estimator_args_ = kwargs
        self.stackingc_ = stackingc

    def _get_subdata(self, X):
        """Returns subsets of the data, one for each class. Assumes the
        columns of X are striped in order.

        e.g. if n_classes_ == 3, then returns (X[:, 0::3], X[:, 1::3],
        X[:, 2::3])

        Parameters
        ----------
        X : np.ndarray, shape=(n, m)
            The feature data.

        Returns
        -------
        array of shape = [len(set(y)), n_samples]
            The subsets of the data.
        """
        if not self.stackingc_:
            return [X, ] * self.n_classes_

        result = []
        for i in range(self.n_classes_):
            slc = (slice(None), slice(i, None, self.n_classes_))
            result.append(X[slc])
        return result

    def fit(self, X, y):
        """ Fit the estimator given predictor(s) X and target y. Assumes the
        columns of X are predictions generated by each predictor on each
        class. Fits one estimator for each class.

        Parameters
        ----------
        X : np.ndarray, shape=(n, m)
            The feature data for which to compute the predicted output.

        y : array of shape = [n_samples]
            The actual outputs (class data).
        """
        self.n_classes_ = len(set(y))
        self.estimators_ = []

        # Generate feature data subsets corresponding to each class.
        X_subs = self._get_subdata(X)

        # Fit an instance of the estimator to each data subset.
        for i in range(self.n_classes_):
            e = self.estimator_(**self.estimator_args_)
            y_i = np.array(list(j == i for j in y))
            X_i = X_subs[i]
            e.fit(X_i, y_i)
            self.estimators_.append(e)

    def predict(self, X):
        """ Predict label values with the fitted estimator on 
        predictor(s) X.

        Returns
        -------
        array of shape = [n_samples]
            The predicted label values of the input samples.
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        """ Predict label probabilities with the fitted estimator 
        on predictor(s) X.

        Returns
        -------
        proba : array of shape = [n_samples]
            The predicted label probabilities of the input samples.
        """
        proba = []

        X_subs = self._get_subdata(X)

        for i in range(self.n_classes_):
            e = self.estimators_[i]
            X_i = X_subs[i]
            pred = e.predict(X_i).reshape(-1, 1)
            proba.append(pred)
        proba = np.hstack(proba)

        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer

        assert_all_finite(proba)

        return proba


class Stacking(object):
    """
    Implements stacking/blending.

    David H. Wolpert (1992). Stacked generalization. Neural Networks,
    5:241-259, Pergamon Press.

    Parameters
    ----------
    meta_estimator : string or callable
        May be one of "best", "vote", "average", or any
        classifier or regressor constructor

    estimators : iterator
        An iterable of estimators; each must support predict_proba()

    cv : iterator
        A cross validation object. Base (level-0) estimators are 
        trained on the training folds, then the meta (level-1) estimator 
        is trained on the testing folds.

    stackingc : bool
        Whether to use StackingC or not. For more information, refer to
        the following paper:
          Seewald A.K.: How to Make Stacking Better and Faster While
          Also Taking Care of an Unknown Weakness, in Sammut C.,
          Hoffmann A. (eds.), Proceedings of the Nineteenth
          International Conference on Machine Learning (ICML 2002),
          Morgan Kaufmann Publishers, pp.554-561, 2002.

    kwargs :
        Arguments passed to instantiate meta_estimator.
    """

    # TODO: Support different features for each estimator.
    # TODO: Support "best", "vote", and "average" for already trained
    #       model.
    # TODO: Allow saving of estimators, so they need not be retrained
    #       when trying new stacking methods.

    def __init__(self, meta_estimator, estimators,
                 cv, raw = False, stackingc=False, proba=True,
                 **kwargs):
        self.estimators_ = estimators
        self.n_estimators_ = len(estimators)
        self.cv_ = cv
        self.stackingc_ = stackingc
        self.proba_ = proba
        self.raw_ = raw

        if stackingc:
            if isinstance(meta_estimator, str) or not issubclass(meta_estimator, RegressorMixin):
                raise Exception('StackingC only works with a regressor.')                
        
        self.meta_estimator_ = meta_estimator(**kwargs)

#        if isinstance(meta_estimator, str):
#            if meta_estimator not in ('best',
#                                      'average',
#                                      'vote'):
#                raise Exception('Invalid meta estimator: {0}'.format(meta_estimator))
#            raise Exception('"{0}" meta estimator not implemented'.format(meta_estimator))
#        elif issubclass(meta_estimator, ClassifierMixin):
#            self.meta_estimator_ = meta_estimator(**kwargs)
#        elif issubclass(meta_estimator, RegressorMixin):
#            self.meta_estimator_ = MRLR(meta_estimator, stackingc, **kwargs)
#        else:
#            raise Exception('Invalid meta estimator: {0}'.format(meta_estimator))

    def _base_estimator_predict(self, e, X):
        """ Predict label values with the specified estimator on 
        predictor(s) X.

        Parameters
        ----------
        e : int
            The estimator object.

        X : np.ndarray, shape=(n, m)
            The feature data for which to compute the predicted outputs.

        Returns
        -------
        pred : np.ndarray, shape=(len(X), 1)
            The mean of the label probabilities predicted by the 
            specified estimator for each fold for each instance X.
        """
        # Generate array for the base-level testing set, which is n x n_folds.
        pred = e.predict(X)
        assert_all_finite(pred)
        return pred

    def _base_estimator_predict_proba(self, e, X):
        """ Predict label probabilities with the specified estimator 
        on predictor(s) X.

        Parameters
        ----------
        e : int
            The estimator object.

        X : np.ndarray, shape=(n, m)
            The feature data for which to compute the predicted outputs.

        Returns
        -------
        pred : np.ndarray, shape=(len(X), 1)
            The mean of the label probabilities predicted by the 
            specified estimator for each fold for each instance X.
        """
        # Generate array for the base-level testing set, which is n x n_folds.
       
        pred = e.predict_proba(X)
        assert_all_finite(pred)
        return pred

    def _make_meta(self, X_array):
        """ Make the feature set for the meta (level-1) estimator.

        Parameters
        ----------
        X : np.ndarray, shape=(n, m)
            The feature data.

        Returns
        -------
        An n x len(self.estimators_) array of meta-level features.
        """
        rows = []
        for index in range(len(self.estimators_)):
            e = self.estimators_[index] 
            X = X_array[index]
            
            if self.proba_:
                # Predict label probabilities                
                pred = self._base_estimator_predict_proba(e, X)
            else:
                # Predict label values
                pred = self._base_estimator_predict(e, X)
            rows.append(pred)
        return np.hstack(rows)

    def fit(self, X_array, y):
        """ Fit the estimator given predictor(s) X and target y.

        Parameters
        ----------
        X : np.ndarray, shape=(n, m)
            The feature data for which to compute the predicted output.

        y : array of shape = [n_samples]
            The actual outputs (class data).
        """
               
        # Build meta data.
        X_meta = [] # meta-level features
        y_meta = [] # meta-level labels

        print 'Training and validating the base (level-0) estimator(s)...'
        print
        for i, (a, b) in enumerate(self.cv_):
            print 'Fold [%s]' % (i)

            X_a_array = [X[a] for X in X_array] # training and validation features
            X_b_array = [X[b] for X in X_array]
            
            y_a, y_b = y[a], y[b] # training and validation labels

            # Fit each base estimator using the training set for the fold.
            for index in range(len(self.estimators_)):
                e = self.estimators_[index]  
                X_a = X_a_array[index]
                print '  Training base (level-0) estimator...',
                e.fit(X_a, y_a)
                X_b = X_b_array[index]
                y_predb = e.predict_proba(X_b)[:, 1]
                # Compute ROC curve and area the curve
                fpr, tpr, thresholds = roc_curve(y[b], y_predb)
                roc_auc = auc(fpr, tpr)
                print 'fold auc : ' ,
                print roc_auc ,
                #print(str(roc_auc))
                print 'done.'

            proba = self._make_meta(X_b_array)
            X_meta.append(proba)
            y_meta.append(y_b)
        print

        X_meta = np.vstack(X_meta)
        if y_meta[0].ndim == 1:
            y_meta = np.hstack(y_meta)
        else:
            y_meta = np.vstack(y_meta)

        # Train meta estimator.
        print 'Training meta (level-1) estimator...',
        self.meta_estimator_.fit(X_meta, y_meta)
        print 'done.'

        # Re-train base estimators on full data.        
        for index in range(len(self.estimators_)):
            e = self.estimators_[index]
            X = X_array[index]        
            print 'Re-training base (level-0) estimator %d on full data...' % (index),
            e.fit(X, y)
            print 'done.'

    def predict(self, X_array):
        """ Predict label values with the fitted estimator on 
        predictor(s) X.

        Returns
        -------
        array of shape = [n_samples]
            The predicted label values of the input samples.
        """
        
        X_meta = self._make_meta(X_array)
        return self.meta_estimator_.predict(X_meta)

    def predict_proba(self, X_array):
        """ Predict label probabilities with the fitted estimator 
        on predictor(s) X.

        Returns
        -------
        array of shape = [n_samples]
            The predicted label probabilities of the input samples.
        """        
        X_meta = self._make_meta(X_array)
        return self.meta_estimator_.predict_proba(X_meta)


class StackingFWL(Stacking):
    """
    Implements Feature-Weighted Linear Stacking.

    Sill, J. and Takacs, G. and Mackey, L. and Lin, D.:
    Feature-weighted linear stacking. Arxiv preprint. 2009.

    """
    pass