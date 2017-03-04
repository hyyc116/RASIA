#coding:utf-8
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
    count = 0
    for file in os.listdir(dirs):
        count+=1
        if count%1000==1:
            print 'progress',count
        filepath=dirs+"/"+file
        xml_str=open(filepath).read()
        soup = parse_xml_with_bs(xml_str)
        for sec in  objects.find_all('sec'):
            if sec.parent.name=='article':
                sec_type = sec.get('sec-type')
                if sec_type is None:
                    sec_type='000'
                sec_header_dic[sec_type]+=1
                title = sec.title.get_text()
                sec_header_dic[title]+=1
    open("sec-type.json","w").write(json.dumps(sec_type_dic))
    open('sec-header.json',"w").write(json.dumps(sec_header_dic))






if __name__=="__main__":
    stat_sec_type(sys.argv[1])


