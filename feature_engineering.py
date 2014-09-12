from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import abc

class Feature_Engineerer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def fit(self, X):        
        "in case we need to fit to external data first we do this as a seperate method"
        return
        
    @abc.abstractmethod
    def transform(self, X):
        "Should take in a pandas dataframe as input, and return a numpy array of engineered features"
        return
        
class NLPEngineer (Feature_Engineerer) :
    def __init__(self,column_name_, max_features_ = 5000):
        self.tfidf =TfidfVectorizer(max_features=max_features_, strip_accents='unicode',  
                analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1,4), use_idf=1,smooth_idf=1,sublinear_tf=1)
        self.vectorized = False  
        self.column_name = column_name_
    
    def fit(self, X):
        print('creating tfidf matrices')  
        self.tfidf.fit(X[self.column_name])
        self.vectorized = True
        return
        
    def transform(self, X):
        if not self.vectorized:
            self.fit(X)        
        return self.tfidf.transform(X[self.column_name]).toarray()
        
class MetadataEngineer(Feature_Engineerer):    
    def __init__ (self) :  
        return
    
    def fit(self, X):        
        return
        
    def transform(self, X):
        #print 'getting metadata features'
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
        utc_difference = (X["unix_timestamp_of_request_utc"] - X["unix_timestamp_of_request"]).as_matrix()
        length_of_post = [len(post) for post in X['request_text_edit_aware']]
        length_of_title = [len(title) for title in X['request_title']]
        timestamps = X["unix_timestamp_of_request"]
        date_times = [datetime.fromtimestamp(ts) for ts in timestamps]
        year = np.array([dt.year for dt in date_times])
        month = np.array([dt.month for dt in date_times])
        enc = OneHotEncoder()
        weekday = np.array([[dt.isocalendar()[2]] for dt in date_times])
        weekday = enc.fit_transform(weekday).toarray()  
        hours =  np.array([[dt.hour] for dt in date_times])        
        hours = enc.fit_transform(hours).toarray()    
        return np.c_[X[features_to_use].as_matrix(), utc_difference,length_of_title,length_of_post, year , month,  weekday]