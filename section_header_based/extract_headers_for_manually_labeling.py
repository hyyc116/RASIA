#coding:utf-8
from bs4 import BeautifulSoup as bs
import re
import sys

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


if __name__=="__main__":
    for line in open(sys.argv[1]):
        path = line.strip()
        doi,headers = parse_xml(path)
        count=0
        for i,header in enumerate(headers):
            count+=1
            sys.stderr.write('{:}\n'.format(count))
            print doi+"\t"+str(i)+"\t"+unicode(header.encode('utf-8'),errors='ignore')+"\t"+path