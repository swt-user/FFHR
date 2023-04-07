"""Hyperbolic operations utils functions."""

import torch

def sq_lorentz_distance(x, v, beta, eval_mode=False):
    """Hyperbolic distance on Poincare balls with varying curvatures c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic queries
        y: torch.Tensor with hyperbolic queries, shape N x d if eval_mode is true else (B x d)
        c: torch.Tensor of size B x 1 with absolute hyperbolic curvatures

    Return: torch,Tensor with hyperbolic distances, size B x 1 if eval_mode is False
            else B x n_entities matrix with all pairs distances
    """
    if eval_mode:
        xv = x @ v.transpose(0,1) 
        res1 = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + beta)   # B x 1
        res2 = torch.sqrt(torch.sum(v * v, dim=-1, keepdim=True).transpose(0,1) + beta)  # B x N
        res = res1 * res2
    else:
        xv = torch.sum(x * v, dim=-1, keepdim=True)
        res = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + beta) * torch.sqrt(torch.sum(v * v, dim=-1, keepdim=True) + beta)   # B x 1
    
    return -2 * (beta + xv - res)
    