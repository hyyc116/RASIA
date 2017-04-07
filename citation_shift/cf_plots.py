#coding:utf-8
import sys
import json
from random import randrange



def load_data(json_path):
    return json.loads(open(json_path).read())

def get_top_N_papers(data,N=10):
    top_dict={}
    for k,v in sorted(data.items(),key=lambda x:x[1]['count_X'],reverse=True):
        top_dict[k]=v

    return top_dict


def random_select_N_papers(data,N=10):
    random_index = random(0,len(data.keys()))
    selected_dic={}
    for index in random_index:
        key = data.keys[index]
        value = data[key]
        selected_dic[key] = value

    return selected_dic

def load_structure_dict(path):
    structure_dict = {}
    for line in open(path):
        line = line.strip()
        splits = line.split(',')
        header = splits[0]
        label = splits[1]
        structure_dict[header]=label

    return structure_dict

# plot count one curve , count X curve
def plot_general_statistics(one_article_dict):
    pass

def plot_structure_dis(one_article_dict):
    pass

def plot_temporal_dis_over_structure(one_article_dict):
    pass


