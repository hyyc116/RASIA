#coding:utf-8
import sys
import json
from random import randrange
from collections import defaultdict


def load_data(json_path):
    return json.loads(open(json_path).read())

def get_top_N_papers_by_countX(data,N=100):
    top_dict={}
    for k,v in sorted(data.items(),key=lambda x:x[1]['count_X'],reverse=True)[:N]:
        top_dict[k]=v

    return top_dict

def get_top_N_papers_by_countO(data,N=100):
    top_dict={}
    for k,v in sorted(data.items(),key=lambda x:len(set(x[1]['count_one'])),reverse=True)[:N]:
        top_dict[k]=v

    return top_dict


def random_select_N_papers(data,N=100):
    # random_index = randrange(0,len(data.keys()))
    selected_dic={}
    for index in range(N):
        key = random.choice(data.keys())
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
def temporal_dis_over_structure(one_article_dict,structure_dict):
    year_heading_dict=defaultdict(dict)
    contexts = one_article_dict['context']
    for ctx in contexts:
        heading = ctx['pos_tit']
        year = ctx['year']
        structure_tag = structure_dict.get(structure_dict,'-1')
        if structure_tag!='-1':
            year_heading_dict[year][structure_tag]=year_heading_dict[year].get(structure_tag,0)+1

    return year_heading_dict

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

#delta_t plot
def cal_citation_delta_t(one_article_dict):
    pass

def main():
    data = load_data('raw_data/plos_cf_ref_dict.json')
    topO_dict = get_top_N_papers_by_countO(data)
    topX_dict = get_top_N_papers_by_countX(data)
    random_dict = random_select_N_papers(data)
    open('plos_topO_100_dict.json','w').write(json.dumps(topO_dict))
    open('plos_topX_100_dict.json','w').write(json.dumps(topX_dict))
    open('plos_rand_100_dict.json','w').write(json.dumps(random_dict))

if __name__ == '__main__':
    main()

