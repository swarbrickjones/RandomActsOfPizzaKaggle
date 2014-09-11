from pandas.io.json import read_json
import numpy as np
from stacking import Stacking
from sklearn.metrics import roc_curve, auc
from nlp import NLPClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn import ensemble
from feature_engineering import NLPEngineer, MetadataEngineer, RawDataClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

#resource.setrlimit(resource.RLIMIT_AS, (10000 * 1048576L, -1L))
def run():
    ### data handling ###       
    print('reading files')
    train_data_raw = read_json('data/train.json')   ## pandas data frame
    test_data_raw = read_json('data/test.json')     ## pandas data frame
     
    ###  train classifiers  ###
    
    #nlp classifier   
    
    print 'getting nlp scores'
    
    y_train = train_data_raw["requester_received_pizza"]     
    nlp_clf_body = RawDataClassifier(NLPClassifier (), NLPEngineer('request_text_edit_aware', max_features_ = 50))
    
    #nlp_clf_title = RawDataClassifier(NLPClassifier (), NLPEngineer('request_title', max_features_ = 1000))
    #nlp_clf_title.fit(train_data_raw, y_train)    
    
    #metadata classifier
    
    print 'getting meta data scores'
    
    gbc = ensemble.GradientBoostingClassifier(n_estimators = 30)
    metadata_clf = RawDataClassifier(gbc, MetadataEngineer())  
    
    estimators = [nlp_clf_body, metadata_clf]
    skf = list(cross_validation.StratifiedKFold(y_train, 10))
    stacking = Stacking(LogisticRegression, estimators,
                 skf, raw = True
                 )
    
    stacking.fit(train_data_raw, y_train)
    
    
#    cv = StratifiedKFold(y_train, n_folds = 10)
#    auc_list = []
#    for i, (train, test) in enumerate(cv):
#        metadata_clf.fit(train_data_raw.iloc[train], y_train[train])
#        nlp_clf_body.fit(train_data_raw.iloc[train], y_train[train])
#        #nlp_clf_title.fit(train_data_raw.iloc[train], y_train[train])
#        proba_meta = metadata_clf.predict_proba(train_data_raw.iloc[test])
#        proba_nlp_body = nlp_clf_body.predict_proba(train_data_raw.iloc[test])
#        #proba_nlp_title = nlp_clf_title.predict_proba(train_data_raw.iloc[test])
#        y_test_pred =np.add( proba_meta[:, 1], proba_nlp_body[:, 1])
#        # Compute ROC curve and area the curve
#        fpr, tpr, thresholds = roc_curve(y_train[test], y_test_pred)
#        roc_auc = auc(fpr, tpr)
#        #print(str(roc_auc))
#        auc_list.append(roc_auc)
#        print(str(roc_auc))
#    cross_fold_mean = np.mean(auc_list)
#    print 'mean auc : ' , cross_fold_mean
    
    nlp_clf_body.fit(train_data_raw, y_train)
    proba_nlp = nlp_clf_body.predict_proba(test_data_raw)
    metadata_clf.fit(train_data_raw, y_train)    
    proba_metadata = metadata_clf.predict_proba(test_data_raw)    
    y_test_pred =np.add( proba_nlp[:, 1], proba_metadata[:, 1])   
    
    test_ids=test_data_raw['request_id']    
    
    fcsv = open('raop_prediction.csv','w')
    fcsv.write("request_id,requester_received_pizza\n")
    for index in range(len(y_test_pred)):
        theline = str(test_ids[index]) + ',' + str(y_test_pred[index])+'\n'
        fcsv.write(theline)
    
    fcsv.close()
    
    ###   word bag scoring   #####
   
    
    
    

if __name__ == '__main__':
    run()