from pandas.io.json import read_json
import numpy as np
from stacking import Stacking
from sklearn.metrics import roc_curve, auc
from nlp import NLPClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn import ensemble
from feature_engineering import NLPEngineer, MetadataEngineer
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
    
    #nlp_clf_title = RawDataClassifier(NLPClassifier (), NLPEngineer('request_title', max_features_ = 1000))
    #nlp_clf_title.fit(train_data_raw, y_train)    
    
    #metadata classifier
    
    print 'getting meta data scores'
    
    meta_clf = ensemble.GradientBoostingClassifier(n_estimators = 30)
    nlp_clf = NLPClassifier ()
    nlp_clf2 = NLPClassifier ()
    estimators = [meta_clf, nlp_clf,nlp_clf2]
 
    meta_engineer = MetadataEngineer()
    X_meta_train = meta_engineer.transform(train_data_raw)
    
    nlp_engineer = NLPEngineer('request_text_edit_aware', max_features_ = 5000)
    X_nlp_train = nlp_engineer.transform(train_data_raw)
    
    nlp_engineer2 = NLPEngineer('request_title', max_features_ = 5000)
    X_nlp_train2 = nlp_engineer2.transform(train_data_raw)
    
    input_train = [X_meta_train,X_nlp_train,X_nlp_train2]
    
    skf = list(cross_validation.StratifiedKFold(y_train, 10))
    stacking = Stacking(LogisticRegression, estimators,
                 skf, raw = True
                 )
    
    stacking.fit(input_train, y_train)
    
    X_meta_test = meta_engineer.transform(test_data_raw)  
    X_nlp_test = nlp_engineer.transform(test_data_raw)
    X_nlp_test2 = nlp_engineer2.transform(test_data_raw)    
    input_test = [X_meta_test,X_nlp_test,X_nlp_test2]    
            
    y_test_pred = stacking.predict_proba(input_test)[:, 1]
    
    test_ids=test_data_raw['request_id']    

    print 'writing to file'    
    
    fcsv = open('raop_prediction.csv','w')
    fcsv.write("request_id,requester_received_pizza\n")
    for index in range(len(y_test_pred)):
        theline = str(test_ids[index]) + ',' + str(y_test_pred[index])+'\n'
        fcsv.write(theline)
    
    fcsv.close()
    
    ###   word bag scoring   #####
   
    
    
    

if __name__ == '__main__':
    run()