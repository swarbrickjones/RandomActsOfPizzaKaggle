from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from  sklearn import linear_model 
import numpy as np

class NLPClassifier (object):
    
    def __init__ (self):       
        self.inner_classifier = linear_model.LogisticRegression()
        
    def fit(self, X_train,y_train) :        
        self.inner_classifier = self.get_best_nlp_classifier(X_train,y_train) 
    
    def predict_proba(self, X_test) : 
        return self.inner_classifier.predict_proba(X_test)

    def predict(self, X_test) : 
        return self.inner_classifier.predicT(X_test)
        
    def get_best_nlp_classifier (self, X_train, y_train): 
        best_C_val = 0.125
#        print 'cross validating to get best linear classifier'
#        scale_factor = 0.9
#        
#        best_C_auc = 0.0
#        for C in [scale_factor**i for i in range(10,30)] :            
#            cv = StratifiedKFold(y_train, n_folds = 10)
#            clf_nlp = linear_model.LogisticRegression( C = C)            
#            auc_list = []
#            for i, (train, test) in enumerate(cv):            
#                proba_nlp = clf_nlp.fit(X_train[train], y_train[train]).predict_proba(X_train[test])
#                # Compute ROC curve and area the curve
#                fpr, tpr, thresholds = roc_curve(y_train[test], proba_nlp[:, 1])
#                roc_auc = auc(fpr, tpr)                
#                #print(str(roc_auc))
#                auc_list.append(roc_auc)
#            cross_fold_mean = np.mean(auc_list)
#            if cross_fold_mean > best_C_auc:
#                best_C_val = C
#                best_C_auc = cross_fold_mean
#            print 'roc score for  C=', str(C), ' : ', str(cross_fold_mean)
#        print 'creating best_model with C = ', best_C_val 
        
        final_model = linear_model.LogisticRegression(C = best_C_val)
        final_model.fit(X_train, y_train)
        return final_model
        