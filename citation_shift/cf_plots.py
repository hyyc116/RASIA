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

#plot the distribution over structure
def plot_structure_dis(one_article_dict):
    pass

#plot the temporal distribution over structure
def plot_temporal_dis_over_structure(one_article_dict):
    pass

#weighted citation network, Count X or weighted fields
def plot_weighted_citation_network():
    pass

#co-citation, co-cited in different structure
def cocitation_network_within_structrue(one_article_dict):
    pass

def plot_sentiment_curve(one_article_dict):
    pass

#context diversity
def citation_context_diversity(one_article_dict):
    pass

def cal_citation_delta_t(one_article_dict):
    pass

def main():
    data = load_data('raw_data/plos_cf_ref_dict.json')
    top_dict = get_top_N_papers(data)
    open('plos_top_10_dict.json','w').write(json.dumps(top_dict))

if __name__ == '__main__':
    main()

