#coding:utf-8
import sys
import xmltodict
import json
from bs4 import BeautifulSoup
import re
import os
from collections import defaultdict
from nltk.tokenize import sent_tokenize
import re

#parse with soup
def parse_xml_with_bs(xml_str):
    return BeautifulSoup(xml_str,'lxml')

#parse
def parse_refs(soup):
    ref_dict=defaultdict(dict)
    ref_list = soup.find_all('ref-list')[0]
    for ref in ref_list.select('ref'):
        # print '===='
        years=ref.select('year')
        titles=ref.select('article-title')
        if len(years)==1 and len(titles)==1:
            rid = ref.get('id')
            year=years[0].get_text()
            title=titles[0].get_text()
            
            ref_dict[rid]['year']=year
            ref_dict[rid]['title']=title

    return ref_dict

def get_token_citation_sents(soup):
    for sec in  soup.find_all('sec'):
        if sec.parent.name=='article':
            title = sec.title.get_text()
            ps = sec.select('p')
            for p in ps:
                # print '==='
                # print p
                last_sent=''
                for sent in sent_tokenize(re.sub(r'\s+',' ',p.prettify()).replace('<p>','').replace('</p>','')):
                    # print '---'
                    if len(sent.split())<15:
                        last_sent+=sent
                    else:
                        if len(last_sent.strip())>15:
                            yield last_sent,title
                            last_sent=sent
                        else:
                            last_sent+=sent

                yield last_sent,title
            # break

def parse_one(path,all_refs_dict):
    xml_str = open('../test/test.XML').read()
    soup = parse_xml_with_bs(xml_str)
    # doi=soup.select()
    ref_dict = parse_refs(soup)
    for sent,title in get_token_citation_sents(soup):
      
        if "<xref" not in sent:
            continue
        # print '=='
        # print sent
        sent_soup = parse_xml_with_bs(sent)
        xrefs = sent_soup.select('xref')
        if len(xrefs)>0:
            sent_str = sent_soup.get_text()
            sent_str = re.sub(r'\s+',' ',sent_str)
            for xref in xrefs:
                # print xref
                if xref.get('ref-type')=='bibr':
                    rid = xref.get('rid')
                    if rid in ref_dict:
                        ref_title = ref_dict[rid]['title']
                        ref_year=ref_dict[rid]['year']
                        label = re.sub(r'\s+'," ",ref_title.lower())+"\t"+ref_year
                        
                        #update dict
                        one_ref_dict = all_refs_dict.get(label,{})
                        one_ref_dict['ref']=ref_dict[rid]
                        countonelist = one_ref_dict.get('count_one',[])
                        countonelist.append(path)
                        one_ref_dict['count_one'] = countonelist
                        one_ref_dict['count_X'] = one_ref_dict.get('count_X',0)+1
                        ctx = one_ref_dict.get('context',[])
                        c = {}
                        c['path']=path
                        c['pos_tit']=title
                        c['context']=sent_str
                        ctx.append(c)
                        one_ref_dict['context']=ctx
                        all_refs_dict[label]=one_ref_dict
    return all_refs_dict

def parse_index(indexfile):
    all_refs_dict={}
    progress=0
    for line in open(indexfile):
        progress+=1
        if progress%100==1:
            print 'PROGRESS',progress
        path = line.strip()
        all_refs_dict = parse_one(path,all_refs_dict)

    print 'done'
    open('raw_data/plos_cf_ref_dict.json','w').write(json.dumps(all_refs_dict))


if __name__ == '__main__':
    all_refs_dict={}
    all_refs_dict = parse_one('../test/test.XML',all_refs_dict)
    for label in all_refs_dict.keys():
        print '==============='
        print label
        print all_refs_dict[label]