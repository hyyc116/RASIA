#coding:utf-8

import fasttext

class fasttext_trainer:

    def __init__(self,modelname):
        self.modelname_ = modelname


    def train(self,traindata):
        self.clf_ = fasttext.supervised(traindata,self.modelname_)


    def test(self,testdata):
        return self.clf_.test(testdata)


    def load_model(self):
        self.clf_ = fasttext.load_model(self.modelname_)


    def predict(self,data):
        return self.clf_.predict(data)




