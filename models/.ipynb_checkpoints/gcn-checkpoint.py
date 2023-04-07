
from utils.hyperbolic import expmap0, logmap0

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
import torch.nn as nn
from utils.hyperbolic import *
from utils.euclidean import *
import torch.nn.functional as F
import torch
import time


class GCNLayer(MessagePassing):
    def __init__(self, edge_index, edge_type, data_type, in_channels, out_channels, num_rels, num_ent, args, act=torch.tanh, head_num=1, device="cuda"):
        super(self.__class__, self).__init__(aggr='add', flow='source_to_target', node_dim=0)

        self.p = args
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.data_type = data_type
        self.act = act
        self.device = device
        self.head_num = head_num
        self.num_rels = num_rels
        self.num_ent = num_ent
        self.init_size = self.p.init_size
        # params for init
        self.drop = torch.nn.Dropout(self.p.gcn_dropout)
        self.dropout = torch.nn.Dropout(0.3)

        self.c = nn.Parameter(torch.ones((1), dtype=self.data_type) , requires_grad = self.p.curvature)
        self.leakyrelu = nn.LeakyReLU(0.2)

        self.att_weight = nn.Parameter(
            self.init_size * torch.randn((self.head_num, 1, self.in_channels), 
            dtype=self.data_type), requires_grad=True)   # (1 x F)
        
        self.att_weight2 = nn.Parameter(
            self.init_size * torch.randn((self.head_num, 1, self.in_channels), 
            dtype=self.data_type), requires_grad=True)   # (1 x F)
        
        self.rel_weight = nn.Parameter(
            2 * torch.rand((self.head_num, self.num_rels+1 , self.in_channels), dtype=self.data_type) - 1.0  ,
            requires_grad=True)   # (num_rels x F)
        
        self.rel_bias = nn.Parameter(
            torch.zeros((self.head_num, self.num_rels+1 , self.in_channels), dtype= self.data_type),
            requires_grad =True)     #(num_rels x F)
        
    def forward(self, x, c):
        
        c = F.softplus(c)
        
        x = expmap0(x, c)
        
        entity_list = []
        
        for i in range(self.head_num):
            
            out = self.propagate(self.edge_index, size=None, x=x, edge_type=self.edge_type, headex=i, c=c)
            entity_list.append(0.5 * logmap0(out, c) )
        
        if self.head_num > 1:
            entity = self._get_mean(entity_list, c)
            return 0.5 * logmap0(entity, c) if self.p.no_act else self.act(0.5 * logmap0(out, c))
        else:
            return entity_list[0] if self.p.no_act else self.act(entity_list[0])

    def message(self, edge_index_i, edge_index_j, x_i, x_j, edge_type, headex, c):
        '''
        edge_index_i : [E]
        x_i: [E, F]
        x_j: [E, F]
        '''
        # c = F.softplus(c)
        
        
        b_j = self.rel_transform(x_j, c, headex)  # (E x F)
        

        # start to compute the attention
        v_j = self._get_attention(edge_index_i, x_i, b_j, headex, c)  # (E x 1)
        
        ans = self._get_aggrweight(edge_index_i, b_j, v_j, c)
        
        

        return  ans
    
    
    def update(self, aggr_out):
        
        return aggr_out
    
    # calculate the transformed feature
    def rel_transform(self, x_j, c, headex):
        rel_weight = torch.index_select(self.rel_weight[headex], 0, self.edge_type)
        rel_bias = expmap0(torch.index_select(self.rel_bias[headex], 0, self.edge_type), c)
    
        b_j = givens_rotations(rel_weight, x_j) # (E, F)
        b_j = mobius_add(rel_bias, b_j, c)    # (E, F)

        return b_j
    
    
    # calculate the attention weight
    def _get_attention(self, edge_index_i, x_i, b_j, headex, c):
        b_i = torch.index_select(b_j, 0 , b_j.size()[0]-self.num_ent + edge_index_i)
            
        v_j = self.leakyrelu(torch.sum(self.att_weight[headex] * b_j, dim=-1 , keepdim=True) + 
                            torch.sum(self.att_weight2[headex] * b_i, dim=-1 , keepdim=True))  # E x 1

        return v_j

    
    
    # calculate the aggregation weight
    def _get_aggrweight(self, edge_index_i, b_j, v_j, c):

        lambda_j = 2/(1 - c * torch.sum(b_j * b_j, dim=-1, keepdim=True))  # E x 1
        denominator = torch.abs(v_j) * (lambda_j - 1)  # E x 1
        denom = torch.zeros(self.num_ent, dtype=self.data_type, device=self.device)
        denom = denom.scatter_add_(0, edge_index_i, denominator.squeeze())   # num_ent
        denom = denom[edge_index_i]   # E
        norm = v_j * lambda_j

        alpha = norm/denom.unsqueeze(-1)
        alpha = self.drop(alpha)
        ans = alpha * b_j

        return ans
    
    def _get_mean(self, entity_list, c):
        
        size = entity_list[0].size()
        norm = torch.zeros((size[0], size[1]), dtype=self.data_type, device=self.device)
        denorm = torch.zeros((size[0], 1), dtype=self.data_type, device=self.device)
        
        for entity in entity_list:
            entity = expmap0(entity, c)
            lambda_j = 2/(1 - c * torch.sum(entity * entity, dim=-1, keepdim=True))  # E x 1
            denorm += (lambda_j - 1)  # E x 1
            norm += lambda_j * entity
            
        return norm/denorm
    
    
    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)

    

        

