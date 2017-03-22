#coding:utf-8
"""SVM, RandomForests based classification based on word feature."""
import json
import sys
sys.path.append('.')
sys.path.append('..')
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
import logging

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.metrics import classification_report, accuracy_score, recall_score

from tools.Stemmer import *
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib
import numpy as np
from optparse import OptionParser
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text.lower())
    stems = stem_tokens(tokens, stemmer)
    return stems

class content_based_classifier:

    def __init__(self,test_size=0.4):
        self.tag_set_=['0','1','2','3','4']
        self.tag_dict_={
            '0':'introduction',
            '1':'related_work',
            '2':'method',
            '3':'result',
            '4':'conclusion'
        }
        self.labels_ =[self.tag_dict_[i] for i in sorted(self.tag_dict_.keys())]
        self.test_size_=test_size
        self.vec_= TfidfVectorizer(stokenizer=tokenize, stop_words='english', max_df=0.5)

    #set dataset and get train text
    def set_dataset(self,data_path):
        data=json.loads(open(data_path).read().strip())
        X=[]
        y=[]
        for sample in data['data']:
            y.append(int(sample['header']))
            X.append(sample['content'])
        self.X_=self.vec_.fit_transform(X)
        self.y_=y
        logging.info('training data loading complete! length:{:}'.format(len(self.X_)))

    #learn feature selection model
    def learn_FS_model(self):
        clf = ExtraTreesClassifier()
        clf.fit(self.X_,self,y_)
        self.fs_model_ = SelectFromModel(clf, prefit=True)
        self.X_ = self.fs_model_.transfrom(self.X_)

    def feature_selection(self,X):
        self.fs_model_.transfrom(X)


    def split_train_test(self):
        self.X_train_,self.X_test_,self.y_train_,self.y_test_=train_test_split(self.X_,self.y_,test_size=self.test_size_,random_state=0)

    #set the classifier used
    def set_classifier(self,clf):
        self.clf_=clf

    #set parameter space
    def set_param_space(self,params_space):
        self.params_space_ = params_space

    def search_hyper_parameter(self,n_iter=50):
      
        rs = RandomizedSearchCV(self.clf_, self.params_space_,
                        cv=cv,
                        verbose=1,
                        n_jobs=-1,
                        n_iter=n_iter)

        rs.fit(self.X_train_, self.y_train_)

        # crf = rs.best_estimator_
        logging.info('best params:{:}'.format(rs.best_params_))
        logging.info('best CV score:{:}'.format(rs.best_score_))
        # logging.info('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

        return rs.best_estimator_

    def save_feature_vector(self,name):
        joblib.dump(self.vec_,'{:}-vec.pkl'.format(name))
        logging.info('feature extraction vector saved to {:}-vec.pkl'.format(name))

    def save_fs_model(self,name):
        joblib.dump(self.fs_model_,'{:}-fsm.pkl'.format(name))
        logging.info('feature selection vector saved to {:}-fsm.pkl'.format(name))


    def save_model(self,clf,name):
        joblib.dump(clf,'{:}-model.pkl'.format(name))
        logging.info('trained model saved to {:}-model.pkl'.format(name))

    # train and test
    def train_and_test(self,params_space,clf):
        logging.info('==== STARTING TO TRAIN MODELS ====')
        logging.info('---- SET PARAMS SPACE ----')
        self.set_param_space(params_space)
        logging.info('---- SET CLASSIFIER ----')
        self.set_classifier(clf)
        logging.info('---- SEARCH HYPER PARAMETERS ON training dataset ----')
        best_clf = self.search_hyper_parameter()  
        logging.info('---- TEST TRAINED MODEL ON testing dataset ----')
        y_pred = best_clf.predict(self.X_test_)
        print(classification_report(self.y_test_,y_pred, target_names=self.labels_, digits=4))
        return best_clf


def train_SVM(datapath,name):
    logging.info('#### train section content based model with SVM ####')
    classifier = content_based_classifier()
    logging.info('==== reading training data ====')
    classifier.set_dataset(datapath)
    logging.info('==== feature selection ====')
    classifier.learn_FS_model()
    classifier.save_fs_model(name)
    logging.info('==== feature selection ====')
    classifier.split_train_test()

    logging.info('##### Initialize SVM ####')
    clf=svm.SVC()
    params_space = {
        'C': scipy.stats.expon(scale=100),
        'gamma': scipy.stats.expon(scale=0.1),
        'kernel':['rbf']
    }
    best_clf = classifier.train_and_test(params_space,clf)
    logging.info('---- saved learned feature vector ----')
    classifier.save_feature_vector(name)
    logging.info('---- saved learned model ----')
    classifier.save_model(best_clf,name)
    logging.info('---- Training complete ----')

if __name__ == '__main__':
    train_SVM(sys.argv[1],sys.argv[2])















