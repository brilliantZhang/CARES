import pickle
import argparse
import numpy as np
from torch_geometric.data import HeteroData
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose1_4', help='diginetica/tmall/Nowplaying')
parser.add_argument('--sample_num', type=int, default=12)
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
elif dataset == "mango":
    num = 149627
else:
    num = 3

relationi = []
relationc = []
neighbor = [] * num

all_test = set()

relation={}
for i in range(len(seq)):
    data = seq[i]
    c = cate[i]
    
    for k in range(1, 4):
        for j in range(len(data)-k):
            # relationi.append([data[j], data[j+k]])
            # relationi.append([data[j+k], data[j]])
            # if c[j]!=c[j+k]:
            #     relationc.append([data[j], data[j+k]])
            #     relationc.append([data[j+k], data[j]])
            if (c[j],c[j+k]) not in relation.keys():
                relation[(c[j],c[j+k])]=[[data[j], data[j+k]]]
                relation[(c[j],c[j+k])]=[[data[j+k], data[j]]]
            else:
                relation[(c[j],c[j+k])].append([data[j], data[j+k]])
                relation[(c[j],c[j+k])].append([data[j+k], data[j]])
                

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

adj1,weight1 = get_adj_weight(relationi,item_dict)
adj2,weight2 = get_adj_weight(relationc,item_dict)

relation1 = [[],[]]
relation2 = [[],[]]
edge_attr1 = []
edge_attr2 = []
for i in range(1,num):
    adj1[i] = adj1[i][:sample_num]
    adj2[i] = adj2[i][:sample_num]
    relation1[0] += [i]*(len(adj1[i]))
    relation1[1] += adj1[i]

    relation2[0] += [i]*(len(adj2[i]))
    relation2[1] += adj2[i]

    edge_attr1 += weight1[i][:sample_num]
    edge_attr2 += weight2[i][:sample_num]

print(len(relation1[0]),len(relation2[0]))

graph = HeteroData()
graph['item'].x=torch.arange(num)
graph[ ('item', 'cate_change', 'item')].edge_index = torch.tensor(relation2, dtype=torch.long)
graph[ ('item', 'neighbor', 'item')].edge_index = torch.tensor(relation1, dtype=torch.long)

graph[ ('item', 'cate_change', 'item')].edge_attr = torch.tensor(edge_attr2, dtype=torch.float)
graph[ ('item', 'neighbor', 'item')].edge_attr = torch.tensor(edge_attr1, dtype=torch.float)

print('Start saving into pyg data\n')
torch.save(graph, dataset + ".pt")
print('Complete saving into pyg data\n')


