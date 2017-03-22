#coding:utf-8
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

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

class headers_based_classifer:
    def __init__(self):
        self.tag_set_=['0','1','2','3','4']
        self.tag_dict_={
            '0':'introduction',
            '1':'related_work',
            '2':'method',
            '3':'result',
            '4':'conclusion'
        }
        self.labels_ =[self.tag_dict_[i] for i in sorted(self.tag_dict_.keys())]


    def set_csv(self,csv_path):
        self.csv_path_ = csv_path


    def crf_features(self,article,i):
        # logging.info('length of article:{:},{:}'.format(len(article),i))
        title = stemsentence(article[i][1].lower()).split()
        if len(title)<2:
            title=title*4
        position = article[i][2]


        features={
            'position':position,
            'relative_position':(int(position)+1)*10/(len(article)),
            'word[0]':title[0],
            'word[-1]':title[-1],
            'word[1]':title[1],
            'word[-2]':title[-2],
            'all':'_'.join(title)
        }

        if i>0:
            title = stemsentence(article[i-1][1].lower()).split()
            if len(title)<2:
                title=title*4
            position = article[i-1][2]
            features.update({
                '-1:position': position,
                '-1:relative_position':(int(position)+1)*10/(len(article)),
                '-1:word[0]':title[0],
                '-1:word[-1]':title[-1]
            })
        else:
            features['BOS']=True

        if i<len(article)-1:
            title = stemsentence(article[i+1][1].lower()).split()
            if len(title)<2:
                title=title*4
            position = article[i+1][2]
            features.update({
                '+1:position':position,
                '+1:relative_position':(int(position)+1)*10/(len(article)),
                '+1:word[0]':title[0],
                '+1:word[-1]':title[-1]
            })
        else:
            features['EOS']=True

        return features

    def parcit_features(self,article,i):
        title = stemsentence(article[i][1].lower()).split()
        if len(title)<2:
            title=title*4
        position = article[i][2]


        features={
            'position':position,
            'relative_position':(int(position)+1)*10/(len(article)),
            'word[0]':title[0],
            'word[1]':title[1],
            'all':'_'.join(title)
        }

        return features


    def articles2features(self,article,feature_func):
        return [feature_func(article,i) for i in range(len(article))]

    def articles2labels(self,article):
        return [label for doi,title,position,label,tag,path in article]

    def article2svm_labels(self,article):
        return [int(tag) for doi,title,position,label,tag,path in article]

    def learn_dict_vec(self,articles):
        self.vec_ = DictVectorizer()
        features=[]
        for article in articles:
            features.extend(self.articles2features(article,self.crf_features))
        
        # print features[0]

        self.vec_.fit(features)
        return self.vec_

    def transform_vec(self,articles):
        features=[]
        for article in articles:
            features.extend(self.articles2features(article,self.crf_features))
        # print features[0]
        return self.vec_.transform(features)


    #read csv data
    def read_data(self):
        article=[]
        last_doi=None
        for line in open(self.csv_path_):
            line = line.strip()
            splits = line.split(',')
            doi = splits[0].strip()
            title = splits[2]
            position = splits[1]
            tag = splits[3]
            path = splits[4]

            if last_doi!=None and len(article)!=0 and doi!=last_doi:
                # print article
                yield article
                article=[]
            
            last_doi=doi
            if tag not in self.tag_set_:
                continue

            article.append((doi,title,position,self.tag_dict_[tag],tag,path))

        if len(article)>3:
            yield article

    #generate dataset
    def get_train_test_data(self,folder=3):
        train_datasets,test_datasets=[],[]
        index=0
        for article in self.read_data():
            index+=1
            if index%folder!=0:
                train_datasets.append(article)
            else:
                test_datasets.append(article)

        logging.info('Size of training data:{:}'.format(len(train_datasets)))
        logging.info('Size of testing data:{:}'.format(len(test_datasets)))
        return train_datasets,test_datasets

    def gen_train_test_Xy(self,data,feature_func):
        train_datasets,test_datasets = data

        self.X_train = [self.articles2features(article,feature_func) for article in train_datasets]
        self.y_train = [self.articles2labels(article) for article in train_datasets]

        #test datasets
        self.X_test = [self.articles2features(article,feature_func) for article in test_datasets]
        self.y_test = [self.articles2labels(article) for article in test_datasets]

    def gen_svm_train_test_Xy(self,data):
        train_dataset,test_dataset = data

        self.X_train = self.transform_vec(train_dataset).toarray()
        self.y_train =[]
        for article in train_dataset:
            self.y_train.extend(self.article2svm_labels(article))
        self.y_train = np.array(self.y_train)
        # print self.X_train[1]
        # print self.y_train[1]
        # for x in self.X_train:
            # print sum(x)


        self.X_test = self.transform_vec(test_dataset).toarray()
        self.y_test =[]
        for article in test_dataset:
            self.y_test.extend(self.article2svm_labels(article))
        self.y_test = np.array(self.y_test)
        # logging.info(train_dataset[:2])


    #set parameter space
    def set_param_space(self,params_space):
        self.params_space_ = params_space

    def set_classifier(self,clf):
        self.clf_=clf

    def train(self):
        self.clf_.fit(self.X_train,self.y_train)


    def search_hyperparameters(self,cv=5,n_iter=80):
        f1_scorer = make_scorer(metrics.flat_f1_score,
                        average='weighted', labels=self.labels_)

        rs = RandomizedSearchCV(self.clf_, self.params_space_,
                        cv=cv,
                        verbose=1,
                        n_jobs=-1,
                        n_iter=n_iter,
                        scoring=f1_scorer)

        rs.fit(self.X_train, self.y_train)

        # crf = rs.best_estimator_
        logging.info('best params:{:}'.format(rs.best_params_))
        logging.info('best CV score:{:}'.format(rs.best_score_))
        # logging.info('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

        return rs.best_estimator_

    def search_svm_hyperparameters(self,cv=5,n_iter=20):
        
        rs = RandomizedSearchCV(self.clf_, self.params_space_,
                        cv=cv,
                        verbose=1,
                        n_jobs=-1,
                        n_iter=n_iter)

        rs.fit(self.X_train, self.y_train)

        # crf = rs.best_estimator_
        logging.info('best params:{:}'.format(rs.best_params_))
        logging.info('best CV score:{:}'.format(rs.best_score_))
        # logging.info('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

        return rs.best_estimator_

    def test(self,clf):
        y_pred = clf.predict(self.X_test)
        print(metrics.flat_classification_report(self.y_test, y_pred, labels=self.labels_, digits=4))

    def testY(self,clf):
        y_pred = clf.predict(self.X_test)
        print(classification_report(self.y_test,y_pred, target_names=self.labels_, digits=4))
        # print(recall_score(self.y_test,y_pred))
        # print self.y_test
        # print y_pred

    def save_crf_model(self,clf,name):
        joblib.dump(clf, '{:}-model.pkl'.format(name))

    def save_svm_model(self,vector,clf,name):
        joblib.dump(clf,'{:}-model.pkl'.format(name))
        joblib.dump(vector,'{:}-vec.pkl'.format(name))

    def load_crf_model(self,path):
        return joblib.load(path)

    def load_svm_model(self,model_path,vec_path):
        model = joblib.load(model_path)
        vec = joblib.load(vec_path)
        return vec,model


    #prediction of new sample
    def predict_X(self,X):
        return self.clf_.predict(X)

    # crf features
    def extract_crf_feature_from_file(self,path):
        article=[]
        features=[]
        for line in open(path):
            line = line.strip()
            if line=='' and len(article)!=0:
                # article
                feature = self.articles2features(article,self.parcit_features)
                features.append(feature)
                article=[]
            else:
                splits = line.split(',')
                doi = splits[0]
                position = splits[1]
                header = splits[2]
                article.append((doi,header,position))

        feature = self.articles2features(article,self.crf_features)
        # print feature
        features.append(feature)


        return features

    # SVM features
    def extract_svm_feature_from_file(self,path):
        article=[]
        features=[]
        for line in open(path):
            line = line.strip()
            if line=='' and len(article)!=0:
                # article
                feature = self.articles2features(article,self.crf_features)
                # print feature
                features.extend(feature)
                article=[]
            else:
                splits = line.split(',')
                doi = splits[0]
                position = splits[1]
                header = splits[2]
                article.append((doi,header,position))

        feature = self.articles2features(article,self.crf_features)
        # print feature
        features.extend(feature)

        # print len(features)

        return features


def prediction_with_SVM(model_path,vec_path,filename):
    classifier = headers_based_classifer()
    vec,model = classifier.load_svm_model(model_path,vec_path)
    classifier.set_classifier(model)
    features = classifier.extract_svm_feature_from_file(filename)
    transform_vec = vec.transform(features).todense()
    print [classifier.tag_dict_[str(tag)] for tag in classifier.predict_X(transform_vec)]

def prediction_with_CRF(model_path,filename):
    classifier = headers_based_classifer()
    model = classifier.load_crf_model(model_path)
    classifier.set_classifier(model)
    features = classifier.extract_crf_feature_from_file(filename)
    print classifier.predict_X(features)


def train_and_test_CRF(train_data_path,name = 'models/crf'):

    classifier = headers_based_classifer()
    feature_func = classifier.parcit_features
    # if name=='Ours':
    #     feature_func = classifier.crf_features
    classifier.set_csv(train_data_path)

    logging.info('#### generating training dataset and test dataset ####')
    data = classifier.get_train_test_data()

    logging.info('#### initialize CRF ####')
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=100,
        all_possible_transitions=True
    )
    classifier.set_classifier(crf)

    logging.info('---- TRAINING WITH parscit FEATURES ----')
    classifier.gen_train_test_Xy(data,feature_func)

    logging.info('---- RANDOMLIZED GRID SEARCH ----')
    params_space = {
        'c1': scipy.stats.expon(scale=0.05),
        'c2': scipy.stats.expon(scale=0.05)
    }
    logging.info('---- set params space ----')
    classifier.set_param_space(params_space)
    best_clf = classifier.search_hyperparameters(n_iter=100)
    classifier.save_crf_model(best_clf,name)
    logging.info('saved trained crf model to {:}-model.pkl'.format(name))

    logging.info('---- TESTING ----')
    classifier.test(best_clf)

def train_and_test_SVM(train_data_path,name = 'models/svm'):
    classifier = headers_based_classifer()
    classifier.set_csv(train_data_path)
    logging.info('#### generating training dataset and test dataset ####')
    data = classifier.get_train_test_data()
    logging.info('##### Initialize SVM ####')
    clf=svm.SVC()
    classifier.set_classifier(clf)

    logging.info('------ Learn Dict Vecotr ----')
    all_dataset=[]
    train_dataset,test_datasets = data
    all_dataset.extend(train_dataset)
    all_dataset.extend(test_datasets)
    dict_vec = classifier.learn_dict_vec(all_dataset)
    
    logging.info('----- generating svm train and test data ----')
    classifier.gen_svm_train_test_Xy(data)

    logging.info('---- Set params space ----')
    params_space = {
        'C': scipy.stats.expon(scale=100),
        'gamma': scipy.stats.expon(scale=0.1),
        'kernel':['rbf']
    }
    classifier.set_param_space(params_space)
    best_clf = classifier.search_svm_hyperparameters(n_iter=100)
    # classifier.train()
    classifier.save_svm_model(dict_vec,best_clf,name)
    logging.info('save trained svm model to {:}-model.pkl and {:}-vec.pkl'.format(name,name))

    logging.info('---- TESTING ----')
    classifier.testY(best_clf)



def main():
    # classifier = headers_based_classifer()
    # classifier.set_csv('../sc_headers.csv')
    # logging.info('===== SECTION HEADERS BASED CLASSIFICATION ======')
    # logging.info('')
    # logging.info('#### generating training dataset and test dataset ####')
    # data = classifier.get_train_test_data()
    # train_and_test_CRF(classifier,data,classifier.parcit_features)
    # # train_and_test_CRF(classifier,data,classifier.crf_features,name='Ours')
    # train_and_test_SVM(classifier,data)
    # classifier.train()
    usage = "usage: python %prog [train OR predict][options]"
    parser = OptionParser(usage)
    
    parser.add_option('-i','--input',dest='input', default=None,help="path to training dataset or predict data.")
    parser.add_option('-m','--modelname',dest='modelname', default=None,help="the name of mode [CRF or SVM]")
    parser.add_option('-o','--output',dest='output',default=None,help="Model path and name to be loaded or saved..")


    (options, args) = parser.parse_args()
    # print len(args)
    mode = args[0]
    modelname = options.modelname
    inputfile = options.input
    output = options.output

    if inputfile is None:
        logging.info('Input data file could not be none.')
        parser.print_help()
        os._exit(0)

    if output is None:
        logging.info('Model saved path could not be none.')
        parser.print_help()
        os._exit(0)

    if modelname is None or modelname not in ['SVM','CRF']:
        logging.info('Model name should be chosen in [CRF, SVM]')
        parser.print_help()
        os._exit(0)

    if mode=='train':
        logging.info("train {:} on dataset {:}".format(modelname,inputfile))
        if modelname=='SVM':
            train_and_test_SVM(inputfile,output)
        elif modelname=='CRF':
            train_and_test_CRF(inputfile,output)

    elif mode=='predict':
        logging.info('predict dataset {:} with model {:}'.format(inputfile,modelname))
        if modelname=='SVM':
            modelpath = output+"-model.pkl"
            vecpath = output+"-vec.pkl"
            prediction_with_SVM(modelpath,vecpath,inputfile)
        elif modelname=='CRF':
            modelpath = output+"-model.pkl"
            prediction_with_CRF(modelpath,inputfile)

    else:
        logging.info("Only train OR predict model")

    if len(args)<1:
        parser.print_help()
        print ""
    


if __name__ == '__main__':
    main()


    



















