import math
import torch
import datetime
import numpy as np
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from gnn import GAT
import torchsnooper
from tqdm import tqdm
import time
from utils import *
from collections import defaultdict
import random
from torch_geometric.data import Data  
from torch_geometric.loader import NeighborLoader
from torchmetrics.functional import hamming_distance


def layer_norm(x):
    ave_x = torch.mean(x, -1).unsqueeze(-1)
    x = x - ave_x
    norm_x = torch.sqrt(torch.sum(x**2, -1)).unsqueeze(-1)
    y = x / norm_x

    return y

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

class CARES(Module):
    def __init__(self, opt, n_node, n_cate):
        super(CARES, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.n_cate = n_cate
        self.tra_n_sess = opt.tra_n_sess
        self.tes_n_sess = opt.tes_n_sess
        self.batch_size = opt.batchSize
        self.scale = opt.scale
        self.device = opt.device

        self.tau =  opt.tau
        self.retrieval_num = opt.retrieval_num
        self.topk = opt.topk
        self.num_relations=opt.num_relations
        self.lmd=opt.lmd
        
        self.item_embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.cate_embedding = nn.Embedding(self.n_cate, self.hidden_size)
        self.pos_embedding = nn.Embedding(opt.recent, self.hidden_size)
        self.len_embedding = nn.Embedding(opt.recent*2, self.hidden_size)
        self.item_embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.rel_embedding = nn.Embedding(opt.num_relations, self.hidden_size)
        
        self.m = opt.m
        self.sess_embedding = torch.zeros((self.retrieval_num, self.m),dtype=torch.int8,requires_grad=False).to(opt.device)
        self.his_labels = torch.randint(2,(self.retrieval_num,)).long().to(opt.device)
        self.sess_hash_matrix = torch.randn((self.hidden_size,self.m),requires_grad=False).to(opt.device)
        
        self.SessionEncoder1 = SessionEncoder(self.hidden_size)
        self.SessionEncoder2 = SessionEncoder(self.hidden_size)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.gnn=GAT(in_channels=self.hidden_size, hidden_channels=self.hidden_size, out_channels=self.hidden_size,
                     num_relations=opt.num_relations,n_layers=opt.step).to(opt.device)
        
        
        self.ce_loss_function = nn.CrossEntropyLoss()
        self.kl_loss_function = nn.KLDivLoss(reduction="batchmean")
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

   
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
        
   
    def get_sess_emb(self, hgraph,alias_inputs, items, mask, cates):
        graph_mask = torch.sign(items)
        hidden = self.item_embedding(items) 
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
                                num_neighbors=[12],
                                shuffle=False, batch_size=len(train_items),num_workers=0)
        
        for batch in batch_loader:
            batch_size = batch['item'].batch_size
            batch_items = batch['item']['x']
            batch = batch.to_homogeneous()
            graph_emb,s_node = self.gnn(self.item_embedding(batch.x.to(self.device)), batch.edge_index.to(self.device), 
            self.rel_embedding((batch.edge_type+1).to(self.device)),batch.edge_attr.to(self.device),batch_size,mean_item, graph_mask)    
        
        get = lambda i: graph_emb[i][alias_inputs[i]]
        g_seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
        global_sess = self.SessionEncoder2(g_seq_hidden,mean_item,s_node,mask)
        
        final_sess =  local_sess + global_sess
            
        return final_sess

    
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
        
        if att_weight!=None:
            log_pred = F.softmax(scores/self.scale,dim=-1).log()
         
            soft_targets = F.one_hot((topk_labels-1).view(-1), num_classes=scores.shape[1]).view(scores.shape[0],self.topk,-1)
            soft_targets = (soft_targets * att_weight.view(soft_targets.shape[0],self.topk,1)).sum(1)
            labels = F.softmax(soft_targets,dim=-1)
            
            loss = loss + self.lmd * self.kl_loss_function(log_pred, labels)
        
        return loss,scores
    
   
    def find_similar_sess(self,target_sess,topk,n_recent):
        recent_sess = self.sess_embedding
        hash_neighbor = recent_sess.unsqueeze(0).repeat(target_sess.shape[0],1,1)
        hash_target = torch.where(torch.matmul(target_sess,self.sess_hash_matrix)>0,1,0).unsqueeze(1)
       
        neg_dist = -(hash_neighbor^hash_target).sum(-1)/hash_target.shape[-1]
        weights,idx = torch.topk(neg_dist, topk, largest=True) 
        sim = nn.Softmax(dim=-1)(weights*self.tau)
        topk_labels = self.his_labels[idx.view(-1)].view(target_sess.shape[0],-1,1)

        return sim,topk_labels
    
    def update_history(self,final_sess,targets):
        final_sess = torch.where(torch.matmul(final_sess.data,self.sess_hash_matrix)>0,1,0).type_as(self.sess_embedding)
                                
        self.sess_embedding = torch.cat((self.sess_embedding[-(self.retrieval_num - final_sess.shape[0]):],
                                                    final_sess))
        
        targets = trans_to_cuda(targets).long()
        self.his_labels = torch.cat((self.his_labels[-(self.retrieval_num - final_sess.shape[0]):],
                                                    targets))

        return True
    
    def forward(self, i,data,hgraph):
        alias_inputs, items, mask, targets,cates,dates = data
        alias_inputs = trans_to_cuda(alias_inputs).long()
        items = trans_to_cuda(items).long()
        mask = trans_to_cuda(mask).long()
        
        cates = trans_to_cuda(cates).long()
        dates = trans_to_cuda(dates).long()
        
        final_sess = self.get_sess_emb(hgraph,alias_inputs, items, mask, cates)
        recent_end_idx = dates.min()
        
        if recent_end_idx<self.retrieval_num:
            self.update_history(final_sess,targets)
            
            return layer_norm(final_sess),targets,None,None
            
        if recent_end_idx>=self.retrieval_num: 
            sim,topk_labels = self.find_similar_sess(final_sess, 
                                                                topk=self.topk,n_recent=self.retrieval_num)
            
            
        self.update_history(final_sess,targets)
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
        
        total_loss = total_loss+loss
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
        
        for score, target in zip(sub_scores, targets.numpy()):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
                
        
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr
