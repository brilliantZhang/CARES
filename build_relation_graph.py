import pickle
import argparse
import numpy as np
from torch_geometric.data import HeteroData
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose1_4', help='diginetica/tmall/Nowplaying')
parser.add_argument('--sample_num', type=int, default=12)
parser.add_argument('--rel_num', type=int, default=10)
opt = parser.parse_args()

dataset = opt.dataset
sample_num = opt.sample_num
seq,cate = pickle.load(open(dataset + '/all_train_seq.txt', 'rb'))

if dataset == 'diginetica':
    num = 43098
    item_dict = pickle.load(open(f'{dataset}_item_dict.pkl', 'rb'))
elif opt.dataset == 'yoochoose1_4':
    num = 37484
    item_dict = pickle.load(open(f'yoochoose_item_dict.pkl', 'rb'))
elif opt.dataset == 'yoochoose1_64':
    num = 37484
    item_dict = pickle.load(open(f'yoochoose_item_dict.pkl', 'rb'))
elif dataset == "tmall":
    num = 40728
    item_dict = pickle.load(open(f'Tmall_item_dict.pkl', 'rb'))
elif dataset == "Nowplaying":
    num = 60417
else:
    num = 3

relation={}
for i in range(len(seq)):
    data = seq[i]
    c = cate[i]
    for k in range(1, 4): 
        for j in range(len(data)-k):
            if (c[j],c[j+k]) not in relation.keys() and (c[j+k],c[j]) not in relation.keys():
                relation[(c[j],c[j+k])]=1
                relation[(c[j],c[j+k])]=1
            elif (c[j],c[j+k]) in relation.keys():
                relation[(c[j],c[j+k])]+=1
                relation[(c[j],c[j+k])]+=1
            else:
                relation[(c[j+k],c[j])]+=1
                relation[(c[j+k],c[j])]+=1
                    
                  
topk_r=sorted(relation.items(),key=lambda x:x[1],reverse=True)[:opt.rel_num ]
topk_r=[r[0] for r in topk_r]

relation={}
relationd = []
relations = []
for i in range(len(seq)):
    data = seq[i]
    c = cate[i]
    for k in range(1, 4):
        for j in range(len(data)-k):
            
            if (c[j],c[j+k]) in topk_r:
                if (c[j],c[j+k]) not in relation.keys() :
                    relation[(c[j],c[j+k])]=[[data[j], data[j+k]]]
                    relation[(c[j],c[j+k])]=[[data[j+k], data[j]]]
                else :
                    relation[(c[j],c[j+k])].append([data[j], data[j+k]])
                    relation[(c[j],c[j+k])].append([data[j+k], data[j]])
            else:
                if c[j]!=c[j+k]:
                    relationd.append([data[j], data[j+k]])
                    relationd.append([data[j+k], data[j]])
                else:
                    relations.append([data[j], data[j+k]])
                    relations.append([data[j+k], data[j]])

relation['same']=relations
relation['drift']=relationd

def get_adj_weight(relation,vid_dict):
    adj1 = [dict() for _ in range(num)]
    adj = [[] for _ in range(num)]
    for tup in relation:
        if tup[1] in adj1[tup[0]].keys():
            adj1[tup[0]][tup[1]] += 1/((np.log(vid_dict[tup[0]]**(0.75))+1)*(np.log(vid_dict[tup[1]]**(0.75))+1))
        else:
            adj1[tup[0]][tup[1]] = 1/((np.log(vid_dict[tup[0]]**(0.75))+1)*(np.log(vid_dict[tup[1]]**(0.75))+1))

    weight = [[] for _ in range(num)]

    for t in range(num):
        x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]
        adj[t] = [v[0] for v in x]
        weight[t] = [v[1] for v in x]
    return adj,weight

relation_value={}
for k,v in relation.items():
    relation_value[k]={}
    adj,weight = get_adj_weight(v,item_dict)
    relation_value[k]['adj']=adj
    relation_value[k]['w']=weight


for k,v in relation_value.items():
    relation_value[k]['edge_type'] = [[],[]]
    relation_value[k]['edge_attr'] = []
    for i in range(1,num):
        relation_value[k]['adj'][i] = v['adj'][i][:sample_num]
        relation_value[k]['edge_type'][0] += [i]*(len(v['adj'][i]))
        relation_value[k]['edge_type'][1] += v['adj'][i]
        relation_value[k]['edge_attr'] += v['w'][i][:sample_num]
        
    print(len(relation_value[k]['edge_type'][0]),len(relation_value[k]['edge_type'][1]))
    
graph = HeteroData()
graph['item'].x=torch.arange(num)
graph['edge_type']=list(relation_value.keys())
for k,v in relation_value.items():
    graph[ ('item', str(k), 'item')].edge_index = torch.tensor(relation_value[k]['edge_type'], dtype=torch.long)
    graph[ ('item', str(k), 'item')].edge_attr = torch.tensor(relation_value[k]['edge_attr'], dtype=torch.float)
print('Start saving into pyg data\n')
torch.save(graph, dataset + ".pt")
print('Complete saving into pyg data\n')