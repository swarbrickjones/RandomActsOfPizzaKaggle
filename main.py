from pandas.io.json import read_json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from  sklearn import linear_model 
from  sklearn import ensemble
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
import numpy as np

#resource.setrlimit(resource.RLIMIT_AS, (10000 * 1048576L, -1L))
def run():
    print('reading training file')
    X_train_json = read_json('data/train.json')
    print('creating tfidf matrices')
    features_to_use = [ "requester_account_age_in_days_at_request", \
                        "requester_days_since_first_post_on_raop_at_request", \
                        "requester_number_of_comments_at_request", \
                        "requester_number_of_comments_in_raop_at_request", \
                        "requester_number_of_posts_at_request", \
                        "requester_number_of_posts_on_raop_at_request", \
                        "requester_number_of_subreddits_at_request", \
                        "requester_upvotes_minus_downvotes_at_request", \
                        "requester_upvotes_plus_downvotes_at_request", \
                        ]
    timestamps = X_train_json["unix_timestamp_of_request_utc"]
    date_times = date_times = [datetime.fromtimestamp(ts) for ts in timestamps]
    year = np.array([dt.year for dt in date_times])
    month = np.array([dt.month for dt in date_times])
    print year.shape
    print month.shape
    X_train = np.c_[X_train_json[features_to_use].as_matrix(), year , month]
    print X_train.shape
    y_train = X_train_json["requester_received_pizza"]
    
    
    classifier = ensemble.GradientBoostingClassifier(n_estimators = 40)
#    cv = StratifiedKFold(y_train, n_folds = 10)
#    auc_list = []
#    for i, (train, test) in enumerate(cv):
#        classifier.fit(X_train[train], y_train[train])
#        probas_ = classifier.predict_proba(X_train[test])
#        # Compute ROC curve and area the curve
#        fpr, tpr, thresholds = roc_curve(y_train[test], probas_[:, 1])
#        roc_auc = auc(fpr, tpr)
#        #print(str(roc_auc))
#        auc_list.append(roc_auc)
#    cross_fold_mean = np.mean(auc_list)
#    print 'mean auc : ' , cross_fold_mean
    
    ####   word bag scoring   #####
    
    
    tfidf =TfidfVectorizer(max_features=5000, strip_accents='unicode',  
            analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1,4), use_idf=1,smooth_idf=1,sublinear_tf=1)
    tfidf.fit(X_train_json['request_text_edit_aware'])
    
    X_nlp = tfidf.transform(X_train_json['request_text_edit_aware'])
    X_nlp = X_nlp.toarray()
    
    print('cross validating models')
#    scale_factor = 0.9
#    best_C_val = 0.125
#    best_C_auc = 0.0
#    for C in [scale_factor**i for i in range(10,30)] :
#        cv = StratifiedKFold(y_train, n_folds = 10)
#        clf_nlp = linear_model.LogisticRegression( C = C)
#        classifier = ensemble.GradientBoostingClassifier(n_estimators = 40)
#        auc_list = []
#        for i, (train, test) in enumerate(cv):            
#            proba_nlp = clf_nlp.fit(X_nlp[train], y_train[train]).predict_proba(X_nlp[test])
#            proba_fields = classifier.fit(X_train[train],y_train[train]).predict_proba(X_train[test])
#            proba =np.add( proba_nlp[:, 1], proba_fields[:, 1])
#            # Compute ROC curve and area the curve
#            fpr, tpr, thresholds = roc_curve(y_train[test], proba)
#            roc_auc = auc(fpr, tpr)
#            print(roc_auc)
#            #print(str(roc_auc))
#            auc_list.append(roc_auc)
#        cross_fold_mean = np.mean(auc_list)
#        if cross_fold_mean > best_C_auc:
#            best_C_val = C
#            best_C_auc = cross_fold_mean
#        print 'roc score for  C=', str(C), ' : ', str(cross_fold_mean)
#    print 'best C : ', best_C_val  
    
    classifier.fit(X_train, y_train)
    clf_nlp = linear_model.LogisticRegression(C = 0.125)
    X_test_json = read_json('data/test.json')
    X_nlp_test = tfidf.transform(X_test_json['request_text_edit_aware'])
    X_nlp_test = X_nlp_test.toarray()
    proba_nlp = clf_nlp.fit(X_nlp, y_train).predict_proba(X_nlp_test)
    
    test_ids = X_test_json['request_id']
    timestamps = X_test_json["unix_timestamp_of_request_utc"]
    date_times = date_times = [datetime.fromtimestamp(ts) for ts in timestamps]
    year = np.array([dt.year for dt in date_times])
    month = np.array([dt.month for dt in date_times])
    X_test = np.c_[X_test_json[features_to_use].as_matrix(), year , month]
    
    clf_nlp = linear_model.LogisticRegression(C = 0.125)    
    
    proba_fields = classifier.predict_proba(X_test)
    y_test_pred =np.add( proba_nlp[:, 1], proba_fields[:, 1])   
    
    fcsv = open('raop_prediction.csv','w')
    fcsv.write("request_id,requester_received_pizza\n")
    for index in range(len(X_test)):
        theline = str(test_ids[index]) + ',' + str(y_test_pred[index])+'\n'
        fcsv.write(theline)
    
    fcsv.close()

if __name__ == '__main__':
    run()