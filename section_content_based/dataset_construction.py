#coding:utf-8
"""
Construct content based identification by using some specific headers.
"""
import sys
from collections import defaultdict

def build_dataset(content_headers_path,sc_index_path):
    header_dict=defaultdict(str)
    for line in open(content_headers_path):
        line=line.strip()
        splits = line.split(",")
        header_dict[splits[0]]=splits[1]

    for line in open(sc_index_path):
        path = line.strip()


