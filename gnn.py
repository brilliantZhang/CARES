#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import math
import torch
import datetime
import numpy as np
from torch import nn
from torch.nn import Module, Parameter, BatchNorm1d
import torch.nn.functional as F
import torchsnooper

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import MessagePassing,GATv2Conv,RGATConv,FiLMConv
from typing import Optional, Tuple, Union
from torch import Tensor
from torch_sparse import SparseTensor, set_diag

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
import numpy as np
import torch
from torch.nn.init import xavier_normal_

# class TuckER(torch.nn.Module):
#     def __init__(self, d1,d2,**kwargs):
#         super(TuckER, self).__init__()

#         # self.E = torch.nn.Embedding(ent_num, d1)
#         # self.R = torch.nn.Embedding(rel_num, d2)
#         self.input_dropout = torch.nn.Dropout(0.3)
#         self.hidden_dropout1 = torch.nn.Dropout(0.3)
#         self.hidden_dropout2 = torch.nn.Dropout(0.3)
#         self.loss = torch.nn.BCELoss()

#         self.bn0 = torch.nn.BatchNorm1d(d1)
#         self.bn1 = torch.nn.BatchNorm1d(d1)
#     #@torchsnooper.snoop()    
#     def forward(self, e1, r, E,W):
#         x = self.bn0(e1)
#         x = self.input_dropout(x)
#         x = x.view(-1, 1, e1.size(1)) #tensor<(80, 1, 256)
#         # e1 batch x r x w  
#         W_mat = torch.mm(r, W.view(r.size(1), -1)) #r x w 
#         W_mat = W_mat.view(-1, e1.size(1), e1.size(1)) #tensor<(80, 256, 256)
#         W_mat = self.hidden_dropout1(W_mat)

#         x = torch.bmm(x, W_mat) 
#         x = x.view(-1, e1.size(1))  #tensor<(80, 256)    
#         x = self.bn1(x)
#         x = self.hidden_dropout2(x)
#         x = torch.mm(x, E.transpose(1,0)) #e1 batch x r x w  x all e2
#         pred = torch.sigmoid(x)
        
#         return pred
    
# class GNNFilm(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_relations, n_layers=2,dropout=0.5):
#         super().__init__()
#         self.dropout = dropout
#         self.hidden_size = hidden_channels
#         self.convs = torch.nn.ModuleList()
#         self.convs.append(FiLMConv(in_channels, hidden_channels, num_relations))
#         for _ in range(n_layers - 1):
#             self.convs.append(FiLMConv(hidden_channels, hidden_channels, num_relations))
#         self.norms = torch.nn.ModuleList()
#         for _ in range(n_layers):
#             self.norms.append(BatchNorm1d(hidden_channels))
#         # self.lin_l = torch.nn.Sequential(OrderedDict([
#         #     ('lin1', Linear(hidden_channels, int(hidden_channels//4), bias=True)),
#         #     ('lrelu', torch.nn.LeakyReLU(0.2)),
#         #     ('lin2', Linear(int(hidden_channels//4),out_channels, bias=True))]))
        
#     def att_out(self, hidden, star_node, graph_mask):
#         sim = torch.matmul(hidden, star_node.unsqueeze(-1)).squeeze()
#         sim = torch.exp(sim)
#         sim = sim * graph_mask.float()
#         sim /= torch.sum(sim, -1).unsqueeze(-1) + 1e-24
#         att_hidden = sim.unsqueeze(-1) * hidden
#         output = torch.sum(att_hidden, 1)
#         return output
        
#     def forward(self, x, edge_index,edge_type,batch_size,star_node,graph_mask):
        
#         for i,(conv, norm) in enumerate(zip(self.convs, self.norms)):
#             x = norm(conv(x, edge_index, edge_type))
#             x = F.dropout(x, p=self.dropout, training=self.training)
            
#             hidden = x[:batch_size].view(star_node.shape[0],-1,self.hidden_size)
#             bs, item_num = hidden.shape[0], hidden.shape[1]
            
#             sim = torch.matmul(hidden, star_node.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.hidden_size)
#             alpha = torch.sigmoid(sim).unsqueeze(-1)
#             star_node_repeat = star_node.repeat(1, item_num).view(bs, item_num, self.hidden_size)
#             hidden = (1-alpha) * hidden +  alpha * star_node_repeat
#             star_node = self.att_out(hidden, star_node, graph_mask)

#             x = torch.cat((hidden.view(-1,self.hidden_size),x[batch_size:]))
#             if i < len(self.convs) - 1:
#                 x = x.relu_()
#                 x = F.dropout(x, p=0.2, training=self.training)
       
#         return hidden
        
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations,edge_dim=256+1, n_layers=2,):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.relu = F.relu
        self.hidden_size = hidden_channels
        self.edge_dim = edge_dim
        self.n_layers=n_layers
        #edge_dim=1
        self.convs.append(GATv2Conv(in_channels, hidden_channels,edge_dim =edge_dim))
        
        for i in range(n_layers - 1):
            self.convs.append(GATv2Conv(hidden_channels, hidden_channels,edge_dim=edge_dim))
        self.linear_transform = nn.Linear(self.hidden_size * n_layers, self.hidden_size, bias=True)
          

    def att_out(self, hidden, star_node, graph_mask):
        sim = torch.matmul(hidden, star_node.unsqueeze(-1)).squeeze()
        sim = torch.exp(sim)
        sim = sim * graph_mask.float()
        sim /= torch.sum(sim, -1).unsqueeze(-1) + 1e-24
        att_hidden = sim.unsqueeze(-1) * hidden
        output = torch.sum(att_hidden, 1)
        return output

    #@torchsnooper.snoop()
    def forward(self, x, edge_index,edge_type,edge_attr,batch_size,star_node,graph_mask):
        edge_attr = torch.cat((edge_type.view(-1,self.edge_dim-1),edge_attr.view(-1,1)),1)
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index,edge_attr) # edge embedding
            
            hidden = x[:batch_size].view(star_node.shape[0],-1,self.hidden_size)
            bs, item_num = hidden.shape[0], hidden.shape[1]
            
            sim = torch.matmul(hidden, star_node.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.hidden_size)
            alpha = torch.sigmoid(sim).unsqueeze(-1)
            star_node_repeat = star_node.repeat(1, item_num).view(bs, item_num, self.hidden_size)
            hidden = (1-alpha) * hidden +  alpha * star_node_repeat
            star_node = self.att_out(hidden, star_node, graph_mask)
            
            x = torch.cat((hidden.view(-1,self.hidden_size),x[batch_size:]))
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.2, training=True)
        
        return hidden,x

def layer_norm(x):
    ave_x = torch.mean(x, -1).unsqueeze(-1)
    x = x - ave_x
    norm_x = torch.sqrt(torch.sum(x**2, -1)).unsqueeze(-1)
    y = x / norm_x

    return y

class Dice(nn.Module):
    """The Dice activation function mentioned in the `DIN paper
    https://arxiv.org/abs/1706.06978`
    """

    def __init__(self, epsilon=1e-3):
        super(Dice, self).__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.randn(1))

    def forward(self, x: torch.Tensor):
        # x: N * num_neurons
        avg = x.mean(dim=1)  # N
        avg = avg.unsqueeze(dim=1)  # N * 1
        var = torch.pow(x - avg, 2) + self.epsilon  # N * num_neurons
        var = var.sum(dim=1).unsqueeze(dim=1)  # N * 1

        ps = (x - avg) / torch.sqrt(var)  # N * 1

        ps = nn.Sigmoid()(ps)  # N * 1
        return ps * x + (1 - ps) * self.alpha * x

class MLP(nn.Module):
    """Multi Layer Perceptron Module, it is the most widely used module for 
    learning feature. Note we default add `BatchNorm1d` and `Activation` 
    `Dropout` for each `Linear` Module.

    Args:
        input dim (int): input size of the first Linear Layer.
        output_layer (bool): whether this MLP module is the output layer. If `True`, then append one Linear(*,1) module. 
        dims (list): output size of Linear Layer (default=[]).
        dropout (float): probability of an element to be zeroed (default = 0.5).
        activation (str): the activation function, support `[sigmoid, relu, prelu, dice, softmax]` (default='relu').

    Shape:
        - Input: `(batch_size, input_dim)`
        - Output: `(batch_size, 1)` or `(batch_size, dims[-1])`
    """

    def __init__(self, input_dim, output_layer=True, dims=None, dropout=0, activation="relu"):
        super().__init__()
        if dims is None:
            dims = []
        layers = list()
        for i_dim in dims:
            layers.append(nn.Linear(input_dim, i_dim))
            layers.append(nn.BatchNorm1d(i_dim))
            layers.append(Dice())
            layers.append(nn.Dropout(p=dropout))
            input_dim = i_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
