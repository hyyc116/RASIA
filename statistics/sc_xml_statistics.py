#coding:utf-8
"""stat section headers"""
import sys
from bs4 import UnicodeDammit
from bs4 import SoupStrainer
from bs4 import BeautifulSoup as bs
import re
from collections import defaultdict
from datetime import datetime
import string
import re

#parse xml 
def parse_xml(path):
    soup = bs(open(path).read(),'lxml')
    if len(soup.select('doi'))==0:
        return None,None
    doi = soup.select('doi')[0].get_text()
    headers=[]
    for node in soup.select('h2.svArticle'):
        header = re.sub(r'\d+\.?\s*','',node.get_text()).strip()
        headers.append(header)
    return doi,headers


def parse_content(path):
    isContent = False
    content=open(path).read().strip()
    soup = bs(open(path).read(),'lxml')
    content_node = soup.select('content')[0]
    isContent=False
    content=''
    header=''
    for child in content_node.children:
        name = child.name
        if name is not None:
            if name=='content_start':
                isContent=True
            elif name=='content_end':
                isContent=False

            if isContent:

                if name=='h2':

                    class_str = child['class'][0]

                    if class_str=='svArticle':
                        if header!='':
                            # content = content.translate(None, string.punctuation)
                            
                            content = re.sub(r'<[^>]*?>','',content)
                            content = re.sub(r'[^\w\s]',' ',content)
                            content = re.sub(r'\s+',' ',content)
                            content = content.replace('{math_begin}','').replace('{math_end}','').replace(u' ','').replace(u' )',')')
                            yield header,content

                        header = re.sub('\d*\.?','',child.get_text()).strip()
                        content=''

                if name=='p':
                    class_str = child['class'][0]
                    if class_str=='svArticle_section':
                        content +=child.get_text()


#parse all xml file of sciendirect 
def parse_all(indexpath):
    header_count=0
    progress_count=0
    headerdict=defaultdict(int)
    doiset=set()
    paths=[]
    sys.stderr.write("time:{:}".format(datetime.now()))
    for line in open(indexpath):
        line=line.strip()
        doi,headers = parse_xml(line)
        if doi is None:
            continue
        #if duplicate 
        if doi in doiset:
            continue
        else:
            doiset.add(doi)
            paths.append(line)
            progress_count+=1
            if progress_count%10==1:
                sys.stderr.write("progress:{:}\n".format(progress_count))
        
        for header in headers:
            header_count+=1
            headerdict[header]+=1

    #print high frequent headers
    high_frequency_count=0
    for k,v in sorted(headerdict.items(),key=lambda x:x[1],reverse=True):
        if v>100:
            high_frequency_count+=v
            print "{:}\t{:}".format(k,v)

    sys.stderr.write("Unique papers:{:}.\nNumber of headers:{:}.\n".format(progress_count,header_count))
    sys.stderr.write("Ratio of high frequency headers: {:.10f}\n".format(high_frequency_count/float(header_count)))
    open("data/paths.txt","w").write("\n".join(paths))
    sys.stderr.write("time:{:}".format(datetime.now()))


if __name__=="__main__":
    # parse_all(sys.argv[1])
    parse_content('../test/0.xml')




