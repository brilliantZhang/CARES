import math
import torch
from torch.nn.init import xavier_normal_
from torch import nn,Tensor
from torch.nn import Module, Parameter, BatchNorm1d
import torch.nn.functional as F
import torchsnooper

from torch_geometric.nn import MessagePassing,GATv2Conv
from typing import Optional, Tuple, Union
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations,edge_dim=256+1, n_layers=2,):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.relu = F.relu
        self.hidden_size = hidden_channels
        self.edge_dim = edge_dim
        self.n_layers=n_layers
        self.convs.append(GATv2Conv(in_channels, hidden_channels,edge_dim =edge_dim))
        
        for i in range(n_layers - 1):
            self.convs.append(GATv2Conv(hidden_channels, hidden_channels,edge_dim=edge_dim))
        self.linear_transform = nn.Linear(self.hidden_size * n_layers, self.hidden_size, bias=True)
          

    def att_out(self, hidden, star_node, graph_mask):
        sim = torch.matmul(hidden, star_node.unsqueeze(-1)).squeeze()
        sim = torch.exp(sim)
        sim = sim * graph_mask.float()
        sim = sim / torch.sum(sim, -1).unsqueeze(-1) + 1e-24
        att_hidden = sim.unsqueeze(-1) * hidden
        output = torch.sum(att_hidden, 1)
        return output

    
    def forward(self, x, edge_index,edge_type,edge_attr,batch_size,s_node,graph_mask):
        edge_attr = torch.cat((edge_type.view(-1,self.edge_dim-1),edge_attr.view(-1,1)),1)
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index,edge_attr) 
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.1, training=self.training)
                
            hidden = x[:batch_size].view(s_node.shape[0],-1,self.hidden_size)
            bs, item_num = hidden.shape[0], hidden.shape[1]
            
            sim = torch.matmul(hidden, s_node.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.hidden_size)
            alpha = torch.sigmoid(sim).unsqueeze(-1)
            s_node_repeat = s_node.repeat(1, item_num).view(bs, item_num, self.hidden_size)
            hidden = (1-alpha) * hidden +  alpha * s_node_repeat
            
            
            if i < len(self.convs) - 1:
                s_node = self.att_out(hidden, s_node, graph_mask)
           
        return hidden,s_node

