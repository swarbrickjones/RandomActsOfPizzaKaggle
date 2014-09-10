from pandas.io.json import read_json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from  sklearn import linear_model 
from  sklearn import ensemble
import numpy as np

#resource.setrlimit(resource.RLIMIT_AS, (10000 * 1048576L, -1L))
print('reading files')
X_train_json = read_json('data/train.json')

print('creating tfidf matrices')
tfidf =TfidfVectorizer(max_features=2000, strip_accents='unicode',  
        analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1,4), use_idf=1,smooth_idf=1,sublinear_tf=1)
tfidf.fit(X_train_json['request_text_edit_aware'])

X_train = tfidf.transform(X_train_json['request_text_edit_aware'])
X_train = X_train.toarray()
y_train = X_train_json["requester_received_pizza"]

print('cross validating models')
scale_factor = 0.5
best_C_val = 0.125
best_C_auc = 0.0
#for C in [scale_factor**i for i in range(-10,10)] :
#    cv = StratifiedKFold(y_train, n_folds = 10)
#    classifier = linear_model.LogisticRegression(C = C)
#    auc_list = []
#    for i, (train, test) in enumerate(cv):
#        probas_ = classifier.fit(X_train[train], y_train[train]).predict_proba(X_train[test])
#        # Compute ROC curve and area the curve
#        fpr, tpr, thresholds = roc_curve(y_train[test], probas_[:, 1])
#        roc_auc = auc(fpr, tpr)
#        #print(str(roc_auc))
#        auc_list.append(roc_auc)
#    cross_fold_mean = np.mean(auc_list)
#    if cross_fold_mean > best_C_auc:
#        best_C_val = C
#        best_C_auc = cross_fold_mean
#    print 'roc score for C = ', str(C), ' : ', str(cross_fold_mean)
#print 'best C : ', best_C_val  
#print 'boosted logression attempt'
classifier = ensemble.RandomForestClassifier(n_estimators = 128, n_jobs = -1)

classifier = linear_model.LogisticRegression(C = best_C_val)
X_test_json = read_json('data/test.json')
X_test = tfidf.transform(X_test_json['request_text_edit_aware'])
X_test = X_test.toarray()
test_ids = X_test_json['request_id']
len(test_ids)

y_test_pred = classifier.fit(X_train, y_train).predict_proba(X_test)[:, 1]


fcsv = open('raop_prediction.csv','w')
fcsv.write("request_id,requester_received_pizza\n")
for index in range(len(X_test)):
    theline = str(test_ids[index]) + ',' + str(y_test_pred[index])+'\n'
    fcsv.write(theline)

fcsv.close()