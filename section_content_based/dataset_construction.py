#coding:utf-8
"""
Construct content based identification by using some specific headers.
"""
import sys
sys.path.append('.')
sys.path.append('..')
from collections import defaultdict
from statistics.sc_xml_statistics import *
import json
import logging


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
    data=[]
    count = 0
    for line in open(sc_index_path):
        count+=1
        if count%1000==1:
            logging.info('PROGRESS:{:}'.format(count))
        path = line.strip()
        for header,content in parse_content(path):
            if header in header_set:
                sample={
                    'header':header,
                    'content':content
                }
                data.append(sample)

    data_json={
        'data':data
    }
    logging.info('results saved to {:}'.format(saved_path))
    open(saved_path,'w').write(json.dumps(data_json))
    logging.info('Done')
    

if __name__ == '__main__':
    build_dataset(sys.argv[1],sys.argv[2],sys.argv[3])

                




