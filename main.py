#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import time
import torch
import pickle
import argparse
import numpy as np
import random
from model import *
from utils import *
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='tmall', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=256, help='hidden state size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.00001]
parser.add_argument('--lr_dc', type=float, default=0.8, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=2, help='gnn propogation steps')
parser.add_argument('--scale', type=float, default=12, help='scale')
parser.add_argument('--recent', type=int, default=10, help='recent number of items')
parser.add_argument('--tao', type=int, default=100, help='tao for softmax')
parser.add_argument('--retrieval_num', type=int, default=1500, help='retrieval num')
parser.add_argument('--topk', type=int, default=30, help='topk session')
parser.add_argument('--m', type=int, default=64, help='m for simhash')
parser.add_argument('--lmd4', type=float, default=1)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument('--device', type=str, default=device)
opt = parser.parse_args()
print(opt)
seed=2022
if torch.cuda.is_available():
    print("gpu cuda is available!")
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
else:
    print("cuda is not available! cpu is available!")
    torch.manual_seed(2022)

patience = 2


def main():
    if opt.dataset == 'diginetica':
        n_node =  43098
        n_cate = 1298
        tra_n_sess = 719470
        tes_n_sess = 60858
        opt.lmd4 = 0.1
    elif opt.dataset == 'yoochoose1_4' :
        n_node =  37484
        n_cate = 341
        tra_n_sess = 5917745
        tes_n_sess = 55898
    elif opt.dataset == 'yoochoose1_64' :
        n_node = 37484
        n_cate = 341
        tra_n_sess = 369859
        tes_n_sess = 55898
        opt.lmd4 = 10
    elif opt.dataset == 'tmall' :
        n_node = 40728
        n_cate = 1672
        tra_n_sess = 351268
        tes_n_sess = 25898
        opt.lmd4 = 5
    else:
        n_node = 310
        
    #wandb.init(config=hyperparameter_defaults, project="yoochoose")
    #wandb.init(config=hyperparameter_defaults, project="tamll")
    #config = wandb.config
    
    opt.tra_n_sess = tra_n_sess
    opt.tes_n_sess = tes_n_sess
    opt.n_node = n_node
    
    train_data = pickle.load(open( opt.dataset + '/train.txt', 'rb'))
    
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open(opt.dataset + '/test.txt', 'rb'))
        
    hgraph = torch.load(opt.dataset+'.pt')
    hgraph = hgraph.pin_memory()
    edge_type=hgraph['edge_type']
    opt.num_relations=len(edge_type)+1
    
    model = trans_to_cuda(SessionGraph(opt, n_node,n_cate))
    
    train_data = Data(train_data, opt,edge_type, 'train')
    test_data = Data(test_data, opt, edge_type,'test')
    
    #wandb.watch(model)
    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epochs):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        
        hit, mrr = train_test(model, train_data, test_data,hgraph,epoch)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        
        metrics = {'hit': hit, 'mrr': mrr}
        #wandb.log(metrics)
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMRR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))
    

if __name__ == '__main__':
    main()
