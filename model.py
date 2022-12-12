#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import math
import torch
import datetime
import numpy as np
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from gnn import layer_norm,GAT
import torchsnooper
from tqdm import tqdm
import time
from utils import *
from collections import defaultdict
import random

from torch_geometric.data import Data  
from torch_geometric.loader import NeighborLoader
from torchmetrics.functional import hamming_distance

class project_mlp(nn.Module):
    def __init__(self, dim=256, projection_size=256, hidden_size=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ELU(inplace=True),
            nn.Linear(hidden_size, projection_size),
        )

    def forward(self, x):
        return self.net(x)

class SessionEncoder(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.hidden_size=hidden_size
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_four = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_five = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        
    def forward(self, seq_hidden,mean_item,mean_cate,mask):
        ht = seq_hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])
        q2 = self.linear_two(mean_item).view(mean_item.shape[0], 1, mean_item.shape[1])
        q3 = self.linear_three(seq_hidden)
        q4 = self.linear_four(mean_cate).view(mean_cate.shape[0], 1, mean_cate.shape[1])
        alpha = self.linear_five(torch.sigmoid(q1 + q2 + q3+ q4))
        a = torch.sum(alpha * seq_hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
     
        sess_emb = self.linear_transform(torch.cat([a, ht], 1))
        return sess_emb

class SessionGraph(Module):
    def __init__(self, opt, n_node, n_cate):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.n_cate = n_cate
        self.tra_n_sess = opt.tra_n_sess
        self.tes_n_sess = opt.tes_n_sess
        self.batch_size = opt.batchSize
        self.scale = opt.scale
        self.sample_num = 12
        self.device = opt.device

        self.tao =  opt.tao
        self.retrieval_num = opt.retrieval_num
        self.topk = opt.topk
        self.alpha = 10
        self.num_relations=opt.num_relations
        
        self.lmd4=opt.lmd4
        
        self.item_embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.cate_embedding = nn.Embedding(self.n_cate, self.hidden_size)
        self.pos_embedding = nn.Embedding(opt.recent, self.hidden_size)
        self.len_embedding = nn.Embedding(opt.recent*2, self.hidden_size)
        self.item_embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.rel_embedding = nn.Embedding(opt.num_relations, self.hidden_size)
        
        self.sess_embedding = torch.randn((self.retrieval_num, self.hidden_size),requires_grad=False).to(opt.device)
        self.his_labels = torch.randint(2,(self.retrieval_num,)).long().to(opt.device)
        self.m = opt.m
        self.sess_hash_matrix = torch.randn((self.hidden_size,self.m),requires_grad=False).to(opt.device)
        
        self.SessionEncoder1 = SessionEncoder(self.hidden_size)
        self.SessionEncoder2 = SessionEncoder(self.hidden_size)
        
        self.gnn=GAT(in_channels=self.hidden_size, hidden_channels=self.hidden_size, out_channels=self.hidden_size,
                     num_relations=opt.num_relations,n_layers=opt.step).to(opt.device)
        
        
        self.w_u_z = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.w_u_r = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.w_u = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.u_u_z = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.u_u_r = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.u_u = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.tanh = nn.Tanh()
        
        self.ce_loss_function = nn.CrossEntropyLoss()
        self.kl_loss_function = nn.KLDivLoss(reduction="batchmean")
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    #@torchsnooper.snoop()
    def add_pos_encoding(self,seq_hidden,mask):
        bs, item_num = seq_hidden.shape[0], seq_hidden.shape[1]
        index = torch.arange(item_num-1,-1,-1).unsqueeze(0)
        pos_index = index.repeat(bs, 1).view(bs, item_num)
        
        pos_index = trans_to_cuda(torch.Tensor(pos_index.float()).long())
        len_d = torch.sum(mask,dim=1,keepdim=True).expand_as(mask)
        pos_index = (pos_index-(item_num-len_d))*mask
        
        pos_hidden = self.pos_embedding(pos_index)
        len_hidden = self.len_embedding(len_d)
        seq_hidden = seq_hidden + pos_hidden + len_hidden
        return seq_hidden


    def ave_pooling(self, hidden, graph_mask):
        length = torch.sum(graph_mask, 1)
        hidden = hidden * graph_mask.unsqueeze(-1).float()
        output = torch.sum(hidden, 1) / length.unsqueeze(-1).float()
        return output
        
    #@torchsnooper.snoop()
    def get_sess_emb(self, hgraph,alias_inputs, items, mask, cates,i):
        graph_mask = torch.sign(items) #第一个为0，对应items和cates
        
        #init
        hidden = self.item_embedding(items) #inputs :unique items,原始itemid
        cate_hidden=self.cate_embedding(cates)
        
        mean_item = self.ave_pooling(hidden, graph_mask)
        mean_cate = self.ave_pooling(cate_hidden, graph_mask)
        
        #local
        get = lambda i: hidden[i][alias_inputs[i]]
        h0 = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        
        seq_hidden = self.add_pos_encoding(h0,mask)
        local_sess = self.SessionEncoder1(seq_hidden,mean_item,mean_cate,mask)

        #global
        relation=hgraph['edge_type']
        train_items = items.view(-1)
        batch_loader = NeighborLoader(hgraph, input_nodes=('item', train_items),
                                num_neighbors=[12,12],
                                shuffle=False, batch_size=len(train_items),num_workers=0)
        
        for batch in batch_loader:
            batch_size = batch['item'].batch_size
            batch_items = batch['item']['x']
            
            batch = batch.to_homogeneous()
            graph_emb,x = self.gnn(self.item_embedding(batch.x.to(self.device)), batch.edge_index.to(self.device), 
            self.rel_embedding((batch.edge_type+1).to(self.device)),batch.edge_attr.to(self.device),batch_size,mean_item, graph_mask)    
        
        
        #sess_encoder
        get = lambda i: graph_emb[i][alias_inputs[i]]
        g_seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        
        global_sess = self.SessionEncoder2(g_seq_hidden,mean_item,mean_cate,mask)
        
        final_sess =  local_sess + global_sess
            
        return final_sess

    #@torchsnooper.snoop()
    def eval_score(self,a):
        b = self.item_embedding.weight[1:]
        b = layer_norm(b)
        scores = torch.matmul(a, b.transpose(1, 0))
        scores *= self.scale
        return scores
    
    def compute_loss(self,a,att_weight,topk_labels,targets):
        b = self.item_embedding.weight[1:]
        b = layer_norm(b)
        scores = torch.matmul(a, b.transpose(1, 0))
        scores *= self.scale

        targets = trans_to_cuda(targets).long()
        loss = self.ce_loss_function(scores, targets - 1)
        
        # if att_weight!=None:
        #     # input should be a distribution in the log space
        #     log_pred = F.softmax(scores/self.scale,dim=-1).log()
         
        #     soft_targets = F.one_hot((topk_labels-1).view(-1), num_classes=scores.shape[1]).view(scores.shape[0],self.topk,-1)
        #     soft_targets = (soft_targets * att_weight.view(soft_targets.shape[0],self.topk,1)).sum(1)
        #     labels = F.softmax(soft_targets,dim=-1)
            
        #     loss += self.lmd4 * self.kl_loss_function(log_pred, labels)
        
        return loss,scores
    
    #@torchsnooper.snoop()
    def find_similar_sess(self,target_sess,topk,n_recent):
        #batch recent sess :n_recent*hidden_size,target_sess：batch*hidden_size
        recent_sess = self.sess_embedding
    
        hash_neighbor = torch.where(torch.matmul(recent_sess,self.sess_hash_matrix)>0,1,0).unsqueeze(0).repeat(target_sess.shape[0],1,1)
        hash_target = torch.where(torch.matmul(target_sess,self.sess_hash_matrix)>0,1,0).unsqueeze(1)
       
        neg_dist = -(hash_neighbor^hash_target).sum(-1)/hash_target.shape[-1]
        weights,idx = torch.topk(neg_dist, topk, largest=True) 
        sim = nn.Softmax(dim=-1)(weights*self.tao)
        
        topk_sess = recent_sess[idx.view(target_sess.shape[0]*topk),:].view(target_sess.shape[0],topk,self.hidden_size)
        topk_labels = self.his_labels[idx.view(-1)].view(target_sess.shape[0],-1,1)

        return topk_sess,sim,topk_labels
    #@torchsnooper.snoop()
    def attention_pooling_sess(self,topk_sess,target_sess,sim):
        seq_length = topk_sess.size(1)
        att_weight = sim.view(-1, seq_length)
        neighbor_sess = (att_weight.unsqueeze(-1) * topk_sess).sum(dim=1)  #(batch_size,emb_dim)

        user_z = torch.sigmoid(self.w_u_z(neighbor_sess)+self.u_u_z(target_sess))
        user_r = torch.sigmoid(self.w_u_r(neighbor_sess) + self.u_u_r(target_sess))
        uw_emb_h = self.tanh(self.w_u(neighbor_sess) + self.u_u(user_r*target_sess))
        sess_final = (1-user_z)*target_sess + user_z * uw_emb_h

        return sess_final,att_weight
    
    def update_history(self,dates,final_sess,targets):
        self.sess_embedding = torch.cat((self.sess_embedding[-(self.retrieval_num - final_sess.shape[0]):],
                                                    final_sess.data))
        
        targets = trans_to_cuda(targets).long()
        self.his_labels = torch.cat((self.his_labels[-(self.retrieval_num - final_sess.shape[0]):],
                                                    targets))

        return True
    
    #@torchsnooper.snoop()
    def forward(self, i,data,hgraph):
        alias_inputs, items, mask, targets,cates,dates = data
        alias_inputs = trans_to_cuda(alias_inputs).long()
        items = trans_to_cuda(items).long()
        mask = trans_to_cuda(mask).long()
        
        cates = trans_to_cuda(cates).long()
        dates = trans_to_cuda(dates).long()
        
        final_sess = self.get_sess_emb(hgraph,alias_inputs, items, mask, cates,i)
        recent_end_idx = dates.min()
        
        if recent_end_idx<self.retrieval_num:
            self.update_history(dates,final_sess,targets)
            
            return layer_norm(final_sess),targets,None,None
            
        if recent_end_idx>=self.retrieval_num: 
            topk_sess,sim,topk_labels = self.find_similar_sess(final_sess, 
                                                                topk=self.topk,n_recent=self.retrieval_num)
            #final_sess,att_weight = self.attention_pooling_sess(topk_sess, final_sess, sim)
            
        self.update_history(dates,final_sess,targets)
        return layer_norm(final_sess), targets,sim,topk_labels


def train_test(model, train_data, test_data,hgraph,epoch):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    hgraph = hgraph.to(model.device,non_blocking=True) 
    model.train()
    total_loss = 0.0
   
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=48, batch_size=model.batch_size,
                                                shuffle=False, pin_memory=True)
    
    for j,data in enumerate(tqdm(train_loader)):
        sess_emb,targets,att_weight,topk_labels = model(j, data, hgraph)
        loss,scores = model.compute_loss(sess_emb,att_weight,topk_labels,targets)
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        
        total_loss += loss
        if j%1000==0:
            print('[%d] Loss: %.4f' % (j, loss.item()))
          
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=48, batch_size=model.batch_size,
                                                shuffle=False, pin_memory=True)
    for j,data in enumerate(tqdm(test_loader)):
        sess_emb,targets,_,_ = model(j, data, hgraph)
        scores = model.eval_score(sess_emb)

        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        mask = data[2]
        for score, target, mask in zip(sub_scores, targets.numpy(), mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
        
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr