#coding:utf8
import sys
import random as rn


#从所有实例中选取每隔类别2000个实例
def randomselect(path,label_num,N,label_index,iscontent=False):
    #选择的行数
    all=[]
    #统计每一个label的数量
    alldic={}
    lines=open(path,"r").readlines()
    upper=len(lines)
    indexes=[i for i in range(upper)]
    i=0
    while len(all)<label_num*N:
        #随机的上线是行数-i
        rnnum=rn.randint(0,upper-i)
        #获得随机的行数
        index=indexes[rnnum]
        #将已随机得到的行数与最后一个数进行替换
        tmp=indexes[upper-i-1]
        indexes[upper-i-1]=indexes[rnnum]
        indexes[rnnum]=tmp
        #取出该行的特征
        line=lines[index].strip()
        label=line.split("\t")[label_index]
        count=alldic.get(label,0)
        i+=1
        #如果该行对应的label的数量小于2000,则计入字典
        if count<label_num:
            all.append(index)
            alldic[label]=count+1
    if not iscontent:
        for index in all:
            print index
    else:
        for index in all:
            print lines[index].strip()


    #for j in range(upper):
     #   if j not in all:
      #      sys.stderr.write(lines[j])

def select_random(path,N):
    lines = open(path).readlines()
    upper=len(lines)
    all_list=[]
    indexes = [i for i in range(upper)]
    i=0
    while len(all_list)<N:
        rnnum = rn.randint(0,upper-i)
        index=indexes[rnnum]
        tmp = indexes[upper-i-1]
        indexes[upper-i-1]=indexes[rnnum]
        indexes[rnnum]=tmp
        line=lines[index].strip()
        i+=1
        all_list.append(line)

    for line in all_list:
        print line


if __name__=="__main__":
    clas=sys.argv[1]
    if clas=="rn":
        select_random(sys.argv[2],int(sys.argv[3]))

    else:
        randomselect(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),sys.argv[5])


