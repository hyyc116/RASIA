#coding:utf-8
"""
Construct content based identification by using some specific headers.
"""
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append('.')
sys.path.append('..')
from collections import defaultdict
from statistics.sc_xml_statistics import *
import json
import logging
import random

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)


def build_dataset(content_headers_path,sc_index_path,saved_path):
    logging.info('Loading pre-defined section headers...')
    header_dict=defaultdict(str)
    header_set=set()
    for line in open(content_headers_path):
        line=line.strip()
        splits = line.split(",")
        header_dict[splits[0]]=splits[1]
        header_set.add(splits[0])

    logging.info('{:} pre-defined section headers loaded complete'.format(len(header_set)))
    
    label_dict=defaultdict(list)
    count = 0
    for line in open(sc_index_path):
        count+=1
        if count%100==1:
            logging.info('PROGRESS:{:}'.format(count))
        path = line.strip()
        for header,content in parse_content(path):
            if header in header_set:
                sample={
                    'header':header_dict[header],
                    'content':content
                }
                label_dict[header_dict[header]].append(sample)
    data=[]           
    logging.info('random select 5000 samples for every label.')
    for label in sorted(label_dict.keys()):
        logging.info('{:}:{:}'.format(label,len(label_dict[label])))
        data.extend(random.sample(label_dict[label],5000))

    random.shuffle(data)  

    data_json={
        'data':data
    }
    logging.info('results saved to {:}'.format(saved_path))
    open(saved_path,'w').write(json.dumps(data_json))
    logging.info('Done')
    

if __name__ == '__main__':
    build_dataset(sys.argv[1],sys.argv[2],sys.argv[3])
    # build_dataset('../data/content_headers.csv','../test/index.txt','data.json')
                




