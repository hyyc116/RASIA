#coding:utf-8
"""stat section headers"""
import sys
from bs4 import UnicodeDammit
from bs4 import SoupStrainer
from bs4 import BeautifulSoup as bs
import re
from collections import defaultdict



#parse xml 
def parse_xml(path):
    soup = bs(open(path).read(),'lxml')
    doi = soup.select('doi')[0].get_text()
    headers=[]
    for node in soup.select('h2.svArticle'):
        header = re.sub(r'\d+\.?\s*','',node.get_text()).strip()
        headers.append(header)
    return doi,headers


#parse all xml file of sciendirect 
def parse_all(indexpath):
    header_count=0
    progress_count=0
    headerdict=defaultdict(int)
    doiset=set()
    for line in open(indexpath):
        line=line.strip()
        doi,headers = parse_xml(line)
        #if duplicate 
        if doi in doiset:
            continue
        else:
            progress_count+=1
            if progress_count%1000==1:
                sys.stderr.write("progress:{:}\n".format(progress_count))
        
        for header in headers:
            header_count+=1
            headerdict['header']+=1

    #print high frequent headers
    high_frequency_count=0
    for k,v in sorted(headerdict.items(),key=lambda x:x[1],reverse=True):
        if v>100:
            high_frequency_count+=1
            print "{:}\t{:}".fromat(k,v)

    sys.stderr.write("Unique papers:{:}.\nNumber of headers:{:}.\n".format(progress_count,header_count))
    sys.stderr.write("Ratio of high frequency headers: {:.10f}\n".format(high_frequency_count/float(header_count)))


if __name__=="__main__":
    parse_all(sys.argv[1])




