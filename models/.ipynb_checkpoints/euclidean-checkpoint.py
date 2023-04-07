"""Euclidean Knowledge Graph embedding models where embeddings are in real space."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.base import KGModel
from utils.euclidean import *

EUC_MODELS = ["TransE", "CP", "MurE", "RotE", "RefE", "AttE", "OTE","RescalE", "RescalE_Du", "TuckER", "TuckER_Decomp", "DistMult", "Complex"]


class BaseE(KGModel):
    """Euclidean Knowledge Graph Embedding models.

    Attributes:
        sim: similarity metric to use (dist for distance and dot for dot product)
    """

    def __init__(self, args, edge_index, edge_type):
        super(BaseE, self).__init__(args, edge_index, edge_type, args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
    
        self.sim = args.distance
        self.u = nn.Parameter(torch.ones(1, args.rank, dtype=self.data_type), requires_grad=args.curvature)


    def get_rhs(self, queries, entity, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return entity, self.bt.weight
        else:
            return entity[queries[:, 2]], self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        if self.sim == "dot":
            if eval_mode:
                score = lhs_e @ rhs_e.transpose(0, 1)
            else:
                score = torch.sum(lhs_e * rhs_e, dim=-1, keepdim=True)
        else:
            # u = F.softplus(self.u)
            score = - euc_sqdistance_SL(lhs_e, rhs_e, eval_mode)
        return score


class RescalE(BaseE):

    def __init__(self, args, edge_index, edge_type):
        super(RescalE, self).__init__(args, edge_index, edge_type)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank * self.rank, sparse=args.sparse)
        
        if args.dtype == 'double':
            self.rel_diag.weight.data = self.rel_diag.weight.data.double()
        
        if args.xvaier:
            nn.init.xavier_uniform_(tensor=self.rel_diag.weight)
        else:
            self.rel_diag.weight.data *= self.init_size
        
    def get_queries(self, queries: torch.Tensor, entity):
        """Compute embedding and biases of queries."""
        rel = self.rel_diag(queries[:,1]).view(-1, self.rank, self.rank)
        lhs_e = torch.bmm(rel, entity[queries[:, 0]].unsqueeze(dim=-1)).squeeze()   # (B, rank)
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases

    def get_factors(self, queries, entity):
        """Computes factors for embeddings' regularization.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor] with embeddings to regularize
        """
        head_e = entity[queries[:, 0]]
        rel_e = self.rel_diag(queries[:, 1]).view(-1, self.rank, self.rank)
        rhs_e = entity[queries[:, 2]]
        return head_e, rel_e, rhs_e
    


class DistMult(BaseE):

    def __init__(self, args, edge_index, edge_type):
        super(DistMult, self).__init__(args, edge_index, edge_type)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank, sparse=args.sparse)
        
        if args.dtype == 'double':
            self.rel_diag.weight.data = self.rel_diag.weight.data.double()
        
        if args.xvaier:
            nn.init.xavier_uniform_(tensor=self.rel_diag.weight)
        else:
            # self.rel_diag.weight.data = 2 * self.rel_diag.weight.data - 1.0
            self.rel_diag.weight.data *= self.init_size
            
    def get_queries(self, queries: torch.Tensor, entity):
        """Compute embedding and biases of queries."""
        rel = self.rel_diag(queries[:,1])
        lhs_e = rel * entity[queries[:, 0]]   # (B, rank)
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases

    def get_factors(self, queries, entity):
        """Computes factors for embeddings' regularization.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor] with embeddings to regularize
        """
        head_e = entity[queries[:, 0]]
        rel_e = self.rel_diag(queries[:, 1])
        rhs_e = entity[queries[:, 2]]
        return head_e, rel_e, rhs_e
    
    
class Complex(BaseE):
    """Euclidean 2x2 Givens rotations"""

    def __init__(self, args, edge_index, edge_type):
        super(Complex, self).__init__(args, edge_index, edge_type)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank, sparse=args.sparse)
        
        if args.dtype == 'double':
            self.rel_diag.weight.data = self.rel_diag.weight.data.double()
        
        if args.xvaier:
            nn.init.xavier_uniform_(tensor=self.rel_diag.weight)
        else:
            self.rel_diag.weight.data = 2 * self.rel_diag.weight.data - 1.0

    def get_queries(self, queries: torch.Tensor, entity):
        """Compute embedding and biases of queries."""
        # lhs_e = givens_rotations(self.rel_diag(queries[:, 1]), self.entity(queries[:, 0])) + self.rel(queries[:, 1])
        lhs_e = givens_rotations_stretch(self.rel_diag(queries[:, 1]), entity[queries[:, 0]])
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases
    
    def get_factors(self, queries, entity):
        """Computes factors for embeddings' regularization.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor] with embeddings to regularize
        """
        head_e = entity[queries[:, 0]]
        rel_e = self.rel_diag(queries[:, 1])
        rhs_e = entity[queries[:, 2]]
        return head_e, rel_e, rhs_e
    