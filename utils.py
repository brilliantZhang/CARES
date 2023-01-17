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

class Data(Dataset):
    def __init__(self, data, opt, edge_type,mode):
        cut_inputs = [input_[-opt.recent:] for input_ in data[0]]
        cut_cates = [input_[-opt.recent:] for input_ in data[2]]
        inputs, mask, len_max,cates = data_masks(cut_inputs,cut_cates,[0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        
        self.length = len(inputs)
        self.cates= np.asarray(cates)
        self.edge_type=edge_type
        if mode=='train':
            self.sesstime = np.asarray([i for i in range(len(data[1]))]) 
        else:
            self.sesstime = np.asarray([i+opt.tra_n_sess for i in range(len(data[1]))])

    def __getitem__(self, i):
        inputs, mask, targets,cates,dates = self.inputs[i], self.mask[i], self.targets[i],\
                                self.cates[i],self.sesstime[i]
        
        max_n_node=self.len_max
        node = np.unique(inputs) 
        d=dict(zip(inputs.tolist(),cates.tolist()))
        
        items = node.tolist() + (max_n_node - len(node)) * [0]
        cates = [d[x] for x in node.tolist()] + (max_n_node - len(node)) * [0]
        alias_inputs = [np.where(node == i)[0][0] for i in inputs]
            
        alias_inputs = torch.tensor(alias_inputs).long()
        items = torch.tensor(items).long()
        mask = torch.tensor(mask).long()
        cates = torch.tensor(cates).long()
        dates = torch.tensor(dates).long()
      
        targets = torch.tensor(targets)
        return [alias_inputs,  items, mask, targets,cates,dates]
    
    def __len__(self):
        return self.length
