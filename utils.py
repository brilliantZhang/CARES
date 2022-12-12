#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
import pickle
from torch_geometric.data import HeteroData
import torch
import time
from torch.utils.data import Dataset
from collections import defaultdict
import random
def data_masks(all_usr_pois, all_usr_cates, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    us_cates = [ucates + item_tail * (len_max - le) for ucates, le in zip(all_usr_cates, us_lens)]
    return us_pois, us_msks, len_max,us_cates

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y,cates = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    valid_set_cates = [cates[s] for s in sidx[n_train:]]
    
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]
    train_set_cates = [cates[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y,train_set_cates), (valid_set_x, valid_set_y,valid_set_cates)

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable
    
# class Data():
#     def __init__(self, data, opt, edge_type,mode):
#         cut_inputs = [input_[-opt.recent:] for input_ in data[0]]
#         cut_cates = [input_[-opt.recent:] for input_ in data[2]]
#         inputs, mask, len_max,cates = data_masks(cut_inputs,cut_cates,[0])
#         self.inputs = np.asarray(inputs)
#         self.mask = np.asarray(mask)
#         self.len_max = len_max
#         self.targets = np.asarray(data[1])
        
#         self.ctargets = np.asarray(data[3])
        
#         self.length = len(inputs)
#         self.cates= np.asarray(cates)
#         self.edge_type=edge_type
#         if mode=='train':
#             self.sesstime = np.asarray([i for i in range(len(data[1]))]) #after data augment sub session id is unique(sorted) 
#         else:
#             self.sesstime = np.asarray([i+opt.tra_n_sess for i in range(len(data[1]))])

#     def generate_batch(self, batch_size):
#         n_batch = int(self.length / batch_size)
#         if self.length % batch_size != 0:
#             n_batch += 1
#         slices = np.split(np.arange(n_batch * batch_size), n_batch)
#         slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
#         return slices

#     def get_slice_gnn(self, i):
#         inputs, mask, targets,cates,dates,ctargets = self.inputs[i], self.mask[i], self.targets[i],\
#                                 self.cates[i],self.sesstime[i],self.ctargets[i]
        
#         items, n_node, alias_inputs,u_cates,rels,trel = [], [], [],[],[],[]
#         for u_input in inputs:
#             n_node.append(len(np.unique(u_input)))
#         max_n_node = np.max(n_node) #去重后最大长度
        
#         # graph_alias_inputs=[]
#         # slice_nodes=np.unique(np.concatenate(inputs))
        
#         for u_input,u_cate,ctar in zip(inputs,cates,ctargets):
#             node = np.unique(u_input) # uniuqe items
#             try:
#                 d=dict(zip(u_input.tolist(),u_cate.tolist()))
#             except:
#                 d=dict(zip(u_input.tolist(),u_cate))
            
#             items.append(node.tolist() + (max_n_node - len(node)) * [0])
#             u_cates.append([d[x] for x in node.tolist()] + (max_n_node - len(node)) * [0])
            
#             alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
#             # graph_alias_inputs.append([np.where(slice_nodes == i)[0][0] for i in u_input])
            
#             c=u_cate
#             rel=[]
#             for j in range(len(c)-1):
#                 flag=True
#                 for i,r in enumerate(self.edge_type):
#                     if len(r)!=2:
#                         continue
#                     if (c[j]==r[0] and c[j+1]==r[1])  or (c[j]==r[1] and c[j+1]==r[0]):        
#                         rel.append(i+1) 
#                         flag=False
#                 if flag:        
#                     if c[j]==c[j+1]:
#                         rel.append(i)
#                     else:
#                         rel.append(i+1)
    
#             rels.append(rel[-self.len_max:] + (self.len_max - len(rel)) * [0])
#             #target type
#             flag=True
#             for i,r in enumerate(self.edge_type):
#                 if len(r)!=2:
#                     continue
#                 if (ctar==r[0] and c[j+1]==r[1])  or (ctar==r[1] and c[j+1]==r[0]):   
#                     trel.append(i+1)
#                     flag=False
#             #print(c,j+1,ctar)
#             if flag:
#                 if ctar==c[j+1]:
#                     trel.append(i)
#                 else:
#                     trel.append(i+1)
        
#         alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
#         #graph_alias_inputs = trans_to_cuda(torch.Tensor(graph_alias_inputs).long())
#         #inputs = trans_to_cuda(torch.Tensor(inputs).long())
#         items = trans_to_cuda(torch.Tensor(items).long())
#         #slice_nodes = trans_to_cuda(torch.Tensor(slice_nodes).long())
#         mask = trans_to_cuda(torch.Tensor(mask).long())
#         u_cates = trans_to_cuda(torch.Tensor(u_cates).long())
#         dates = trans_to_cuda(torch.Tensor(dates).long())
#         rels =  trans_to_cuda(torch.Tensor(rels).long())
#         return alias_inputs,  items, mask, targets,u_cates,dates,rels,trel

class Data(Dataset):
    def __init__(self, data, opt, edge_type,mode):
        cut_inputs = [input_[-opt.recent:] for input_ in data[0]]
        cut_cates = [input_[-opt.recent:] for input_ in data[2]]
        inputs, mask, len_max,cates = data_masks(cut_inputs,cut_cates,[0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        
        self.ctargets = np.asarray(data[3])
        
        self.length = len(inputs)
        self.cates= np.asarray(cates)
        self.edge_type=edge_type
        if mode=='train':
            self.sesstime = np.asarray([i for i in range(len(data[1]))]) #after data augment sub session id is unique(sorted) 
        else:
            self.sesstime = np.asarray([i+opt.tra_n_sess for i in range(len(data[1]))])

    # def generate_batch(self, batch_size):
    #     n_batch = int(self.length / batch_size)
    #     if self.length % batch_size != 0:
    #         n_batch += 1
    #     slices = np.split(np.arange(n_batch * batch_size), n_batch)
    #     slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
    #     return slices

    def __getitem__(self, i):
        inputs, mask, targets,cates,dates,ctargets = self.inputs[i], self.mask[i], self.targets[i],\
                                self.cates[i],self.sesstime[i],self.ctargets[i]
        
        # for u_input in inputs:
        #     n_node.append(len(np.unique(u_input)))
        # max_n_node = np.max(n_node) #去重后最大长度
        
        max_n_node=self.len_max
        
        node = np.unique(inputs) # uniuqe items
        try:
            d=dict(zip(inputs.tolist(),cates.tolist()))
        except:
            d=dict(zip(inputs.tolist(),cates))
        
        items = node.tolist() + (max_n_node - len(node)) * [0]
        cates = [d[x] for x in node.tolist()] + (max_n_node - len(node)) * [0]
        
        alias_inputs = [np.where(node == i)[0][0] for i in inputs]
        
        # c=cates
        # rel=[]
        # for j in range(len(c)-1):
        #     flag=True
        #     for i,r in enumerate(self.edge_type):
        #         if len(r)!=2:
        #             continue
        #         if (c[j]==r[0] and c[j+1]==r[1])  or (c[j]==r[1] and c[j+1]==r[0]):        
        #             rel.append(i+1) 
        #             flag=False
        #     if flag:        
        #         if c[j]==c[j+1]:
        #             rel.append(i)
        #         else:
        #             rel.append(i+1)

        # rels=rel[-self.len_max:] + (self.len_max - len(rel)) * [0]
            
        alias_inputs = torch.tensor(alias_inputs).long()
        items = torch.tensor(items).long()
        mask = torch.tensor(mask).long()
        cates = torch.tensor(cates).long()
        dates = torch.tensor(dates).long()
      
        targets = torch.tensor(targets)
        return [alias_inputs,  items, mask, targets,cates,dates]
    
    def __len__(self):
        return self.length
