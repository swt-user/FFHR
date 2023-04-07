"""Hyperbolic Knowledge Graph embedding models where all parameters are defined in tangent spaces."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.base import KGModel
from utils.euclidean import *
from utils.hyperbolic import *

HYP_MODELS = ["MurH", "AttH","RescalH", "ComplexH", "DistMultH", "DualH"]


class BaseH(KGModel):
    """Trainable curvature for each relationship."""

    def __init__(self, args, edge_index, edge_type):
        super(BaseH, self).__init__(args, edge_index, edge_type, args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        
        #self.c = nn.Parameter(torch.ones((1), dtype=self.data_type), requires_grad=args.curvature)

        self.c = nn.Parameter(torch.Tensor([-args.init_c]), requires_grad=args.curvature)

    def get_rhs(self, queries, entity, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return entity, self.bt.weight
        else:
            return entity[queries[:, 2]], self.bt(queries[:, 2])

class MurH(BaseH):
    """Hyperbolic 2x2 Givens rotations"""
    def __init__(self, args, edge_index, edge_type):
        super(MurH, self).__init__(args, edge_index, edge_type)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0

    def get_queries(self, queries, entity):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c)
        head = expmap0(self.rel_diag(queries[:, 1]) * self.entity(queries[:, 0]), c)
        rel = expmap0(self.rel(queries[:,1]), c )

        res = project(mobius_add(head, rel, c), c)

        return (res, c), self.bh(queries[:, 0])
    
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

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e, c = lhs_e
        return - hyp_distance_multi_c(lhs_e, rhs_e, c, eval_mode) ** 2

class AttH(BaseH):
    """Hyperbolic attention model combining translations, reflections and rotations"""

    def __init__(self, args, edge_index, edge_type):
        super(AttH, self).__init__(args, edge_index, edge_type)
        self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], 2 * self.rank), dtype=self.data_type) - 1.0
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.act = nn.Softmax(dim=1)
        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().cuda()
        else:
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    def get_queries(self, queries, entity):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c)
        head = self.entity(queries[:, 0])
        rot_mat, ref_mat = torch.chunk(self.rel_diag(queries[:, 1]), 2, dim=1)
        rot_q = givens_rotations(rot_mat, head).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head).view((-1, 1, self.rank))
        cands = torch.cat([ref_q, rot_q], dim=1)
        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        lhs = expmap0(att_q, c)
        rel = expmap0(self.rel(queries[:, 1]), c)
        res = project(mobius_add(lhs, rel, c), c)
        return (res, c), self.bh(queries[:, 0])

    def get_factors(self, queries, entity):
        """Computes factors for embeddings' regularization.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor] with embeddings to regularize
        """
        head_e = entity[queries[:, 0]]
        rel_e = self.rel_diag(queries[:, 1]) * 0
        rhs_e = entity[queries[:, 2]]
        return head_e, rel_e, rhs_e

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e, c = lhs_e
        return - hyp_distance_multi_c(lhs_e, rhs_e, c, eval_mode) ** 2

        
class DistMultH(BaseH):
    """Hyperbolic 2x2 Givens rotations"""
    def __init__(self, args, edge_index, edge_type):
        super(DistMultH, self).__init__(args, edge_index, edge_type)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank, sparse=args.sparse)
        
        if args.dtype == 'double':
            self.rel_diag.weight.data = self.rel_diag.weight.data.double()
        
        if args.xvaier:
            nn.init.xavier_uniform_(tensor=self.rel_diag.weight)
        else:
            # self.rel_diag.weight.data = 2 * self.rel_diag.weight.data - 1.0
            self.rel_diag.weight.data *= self.init_size
        

    def get_queries(self, queries, entity):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c)
        head = entity[queries[:, 0]]
        res1 = self.rel_diag(queries[:, 1]) * head
        return (res1, c), self.bh(queries[:, 0])
    
    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e, c = lhs_e
        return dot_multi_c(lhs_e, rhs_e, c, eval_mode)
    
    def get_factors(self, queries, entity):
        """Computes factors for embeddings' regularization.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor] with embeddings to regularize
        """
        c = F.softplus(self.c)
        head_e = entity[queries[:, 0]]
        rel_e = self.rel_diag(queries[:, 1]) 
        rhs_e = entity[queries[:, 2]]
        return head_e, rel_e, rhs_e
    
        
class ComplexH(BaseH):
    """Hyperbolic 2x2 Givens rotations"""
    def __init__(self, args, edge_index, edge_type):
        super(ComplexH, self).__init__(args, edge_index, edge_type)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank, sparse=args.sparse)
        
        if args.dtype == 'double':
            self.rel_diag.weight.data = self.rel_diag.weight.data.double()
        
        if args.xvaier:
            nn.init.xavier_uniform_(tensor=self.rel_diag.weight)
        else:
            self.rel_diag.weight.data = 2 * self.rel_diag.weight.data - 1.0
            

    def get_queries(self, queries, entity):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c)
        head = entity[queries[:, 0]]
        res1 = givens_rotations_stretch(self.rel_diag(queries[:, 1]), head)
        return (res1, c), self.bh(queries[:, 0])
    
    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e, c = lhs_e
        return dot_multi_c(lhs_e, rhs_e, c, eval_mode)
    
    def get_factors(self, queries, entity):
        """Computes factors for embeddings' regularization.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor] with embeddings to regularize
        """
        c = F.softplus(self.c)
        head_e = entity[queries[:, 0]]
        rel_e = self.rel_diag(queries[:, 1])
        rhs_e = entity[queries[:, 2]]
        return head_e, rel_e, rhs_e
    
        
class RescalH(BaseH):
    """Hyperbolic 2x2 Givens rotations"""
    def __init__(self, args, edge_index, edge_type):
        super(RescalH, self).__init__(args, edge_index, edge_type)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank * self.rank, sparse=args.sparse)
        
        if args.dtype == 'double':
            self.rel_diag.weight.data = self.rel_diag.weight.data.double()
        
        if args.xvaier:
            nn.init.xavier_uniform_(tensor=self.rel_diag.weight)
        else:
            self.rel_diag.weight.data *= self.init_size
            
    def get_queries(self, queries, entity):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c)
        rel = self.rel_diag(queries[:,1]).view(-1, self.rank, self.rank)
        lhs_e = torch.bmm(rel, entity[queries[:, 0]].unsqueeze(dim=-1)).squeeze()   # (B, rank)
        return (lhs_e, c), self.bh(queries[:, 0])
    
    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e, c = lhs_e
        return dot_multi_c(lhs_e, rhs_e, c, eval_mode)
    
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


class DualH(BaseH):
    """Hyperbolic 2x2 Givens rotations"""
    def __init__(self, args, edge_index, edge_type):
        super(DualH, self).__init__(args, edge_index, edge_type)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank, sparse=args.sparse)
        
        if args.dtype == 'double':
            self.rel_diag.weight.data = self.rel_diag.weight.data.double()
        
        if args.xvaier:
            nn.init.xavier_uniform_(tensor=self.rel_diag.weight)
        else:
            self.rel_diag.weight.data *= self.init_size
            
    def get_queries(self, queries, entity):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c)
        rel = self.rel_diag(queries[:,1])
        lhs_e = given_8_rotation(rel, entity[queries[:, 0]])   # (B, rank)
        return (lhs_e, c), self.bh(queries[:, 0])
    
    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e, c = lhs_e
        return dot_multi_c(lhs_e, rhs_e, c, eval_mode)
    
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
