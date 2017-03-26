#coding:utf-8
import sys
import fasttext
import json
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

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


def train_fasttext(data_path,name):
    logging.info('Generating training data to pre-defined format')
    data=json.loads(open(data_path).read().strip())
    X=[]
    y=[]
    label_dict=defaultdict(int)
    lines=[]
    for sample in data['data']:
        label = sample['header']
        content = sample['content']
        lines.append('__label__{:} {:}'.format(label,content))
        label_dict[label]+=1

    for label in sorted(label_dict.keys()):
        logging.info('{:}:{:}'.format(label,label_dict[label]))

    logging.info('split training dataset')
    train_lines,test_lines = train_test_split(lines,test_size=0.4,random_state=0)
    open('../raw_data/train_data_for_fasttext.txt','w').write('\n'.join(train_lines))
    open('../raw_data/test_data_for_fasttext.txt','w').write('\n'.join(test_lines))

    logging.info('training')

    ft_trainer = fasttext_trainer(name)
    ft_trainer.train('../raw_data/train_data_for_fasttext.txt')
    result = tf_trainer.test('../raw_data/testdata_for_fasttext.txt')
    logging.info('Precision:{:}'.format(result.precision))
    logging.info('Recall:{:}'.format(result.recall))
    logging.info('Number of examples:{:}'.format(result.nexamples))

if __name__=="__main__":
    train_fasttext(sys.argv[1],sys.argv[2])














