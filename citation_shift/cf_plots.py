#coding:utf-8
import sys
import json
import random 
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_data(json_path):
    return json.loads(open(json_path).read())

def get_top_N_papers_by_countX(data,N=100):
    top_dict={}
    for k,v in sorted(data.items(),key=lambda x:x[1]['count_X'],reverse=True)[:N]:
        top_dict[k]=v

    return top_dict

def get_top_N_papers_by_countO(data,N=100):
    top_dict={}
    for k,v in sorted(data.items(),key=lambda x:len(set([one[0] for one in x[1]['count_one']])),reverse=True)[:N]:
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
def get_general_statistics(one_article_dict):
    title = one_article_dict['ref']['title']
    year = int(one_article_dict['ref']['year'])

    count_one_list = one_article_dict['count_one']
    
    count_one_dict = defaultdict(int)
    count_X_dict=defaultdict(int)
    already_set=set()
    for citation in count_one_dict:
        citation_path = citation[0]
        citation_year = int(citation[1])

        count_X_dict[citation_year]+=1
        if citation_path not in already_set:
            count_one_dict[citation_year]+=1

        already_set.add(citation_path)

    return count_one_dict,count_X_dict


#plot the distribution over structure
def get_structure_dis(one_article_dict,structure_dict):
    context_list = one_article_dict['context']

    pos_counter=defaultdict(int)
    for citation in context_list:
        header = citation['pos_tit'].strip()
        context = citation['context']
        year = int(citation['year'])
        pos = structure_dict.get(header,'-1')
        pos_counter[pos]+=1
    
    return pos_counter

#plot the temporal distribution over structure
def get_temporal_structure_dis(one_article_dict,structure_dict):
    year_heading_dict=defaultdict(dict)
    contexts = one_article_dict['context']
    for ctx in contexts:
        heading = ctx['pos_tit']
        year = int(ctx['year'])
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
    open('raw_data/plos_topO_100_dict.json','w').write(json.dumps(topO_dict))
    open('raw_data/plos_topX_100_dict.json','w').write(json.dumps(topX_dict))
    open('raw_data/plos_rand_100_dict.json','w').write(json.dumps(random_dict))

#plot top 10 papers
def plot_top_10(path,structure_path):
    structure_dict = load_structure_dict(structure_path)
    data = json.loads(open(path).read())
    fig,axes = plt.subplots(10,4,figsize(10,50))
    count=0
    for k,v in sorted(data.items(),key=lambda x:x[1]['count_X'],reverse=True)[:10]:
        one_article_dict = v
        count_one_dict,count_X_dict = get_general_statistics(one_article_dict)
        structure_dis = get_structure_dis(one_article_dict,structure_dict)
        temporal_structure_dis = get_temporal_structure_dis(one_article_dict,structure_dict)

        #for count one
        xs=[]
        ys=[]
        for year in sorted(count_one_dict.keys()):
            xs.append(xs)
            ys.append(ys)

        plot_line(xs,ys,ax[count,0],title='Count One')
        #for count X
        xs=[]
        ys=[]
        for year in sorted(count_X_dict.keys()):
            xs.append(xs)
            ys.append(ys)

        plot_line(xs,ys,ax[count,1],title='Count X')

        #plot structure dis
        xs=[]
        ys=[]
        for structure in sorted(structure_dis):
            if structure !='-1':
                xs.append(structure)
                ys.append(structure_dis[ys])

        ys = [float(i)/sum(ys) for i in ys]
        plot_bar(x,y,axes[count,2],title='Distribution Over structure')

        #plot temporal distribution
        temporal_structure_dis = defaultdict(list)
        xs=[]
        for year in sorted(temporal_structure_dis.keys()):
            xs.append(year)
            structure_dis = temporal_structure_dis[year]
            ys_count=0
            for structure in sorted(structure_dis.keys()):
                ys_count+=structure_dis[]

            for structure in sorted(structure_dis.keys()):
                temporal_structure_dis[structure].append(temporal_structure_dis[structure]/float(ys_count))

        ax4= axes[count,3]
        for structure in sorted(temporal_structure_dis.keys()):
            ax4.plot(xs,temporal_structure_dis[structure],label='{:}'.format(structure))

        ax4.legend()
        ax4.set_title('Temporal Structure Distribution')
        count+=1

    plt.tight_layout()
    plt.savefig('raw_data/top_10_figure.png',dpi=300)


def plot_line(x,y,ax,title):
    ax.plot(x,y)
    ax.set_title(title)

def plot_bar(x,y,ax,title):
    x=np.array(x)
    x_pos = np.arange(len(x))
    ax.bar(x_pos,y,align='center',color=cml.get_color('bar'))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x)
    ax.set_title(title)



if __name__ == '__main__':
    # main()
    plot_top_10(sys.argv[1],sys.argv[2])

