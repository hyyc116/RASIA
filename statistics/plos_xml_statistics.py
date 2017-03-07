#coding:utf-8
""" statistic of all plos xml of headers and sec type """
import sys
import xmltodict
import json
from bs4 import BeautifulSoup
import re
import os
from collections import defaultdict


# parse xml
def parse_xml_with_xmltodict(xml_str):
    return json.loads(json.dumps(xmltodict.parse(xml_str)))

#parse with soup
def parse_xml_with_bs(xml_str):
    return BeautifulSoup(xml_str,'lxml')

def stat_sec_type(dirs):
    sec_type_dic=defaultdict(int)
    sec_header_dic=defaultdict(int)
    count=0
    title_count=0
    non_count=0
    for file in os.listdir(dirs):
        count+=1
        if count%1000==1:
            print 'progress',count
        filepath=dirs+"/"+file
        xml_str=open(filepath).read()
        soup = parse_xml_with_bs(xml_str)
        for sec in  soup.find_all('sec'):
            if sec.parent.name=='article':
                sec_type = sec.get('sec-type')
                if sec_type is None:
                    sec_type='000'
                    non_count+=1
                sec_type_dic[sec_type]+=1
                title = sec.title.get_text()
                title_count+=1
                sec_header_dic[title]+=1
    open("data/sec-type.json","w").write(json.dumps(sec_type_dic))
    open('data/sec-header.json',"w").write(json.dumps(sec_header_dic))
    print 'Papers:',count
    print 'Number of section headers:',title_count
    print 'Number without sec type:',non_count
    print 'Ratio of headers with sec type',float(title_count-non_count)/title_count

def get_top_n_sec():
    sec_header_dic = json.loads(open('data/sec-header.json').read())
    sec_type_dic = json.loads(open('data/sec-type.json').read())
    sys.stderr.write('====HEADERS\n')
    for k,v in sorted(sec_header_dic.items(),key=lambda x:x[1], reverse=True):
        if v>100:
            sys.stderr.write('{:}\t{:}\n'.format(k,v))


    sys.stderr.write('====TYPES\n')
    for k,v in sorted(sec_type_dic.items(),key=lambda x:x[1], reverse=True):
        if v>100:
            sys.stderr.write('{:}\t{:}\n'.format(k,v))









if __name__=="__main__":
    stat_sec_type(sys.argv[1])
    get_top_n_sec()


