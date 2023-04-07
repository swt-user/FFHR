# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from abc import ABC, abstractmethod
from typing import Tuple
from utils.euclidean import *
from utils.hyperbolic import *
import torch
from torch import nn
from models.euclidean import EUC_MODELS
from models.hyperbolic import HYP_MODELS


all_regularizers = ["N3", "F2", "DURA_W", "Or_Regu","Or_Regu_double","Or_Regu_F2","DURA_RESCAL", "DURA_RESCAL_H","DURA_ComplexH", "DURA_DistMultH", "DURA_RESCAL_W", "DURA_RESCAL_P", "DURA_DistMultH_P","DURA_RESCAL_1","DURA_RESCAL_2"]


class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass


class F2(Regularizer):
    def __init__(self, weight: float):
        super(F2, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(f ** 2)
        return norm / factors[0].shape[0]


class N3(Regularizer):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        """Regularized complex embeddings https://arxiv.org/pdf/1806.07297.pdf"""
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(
                torch.abs(f) ** 3
            )
        return norm / factors[0].shape[0]

class DURA_W(Regularizer):
    def __init__(self, weight: float):
        super(DURA_W, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
       
        h, r, t = factors

        norm += 0.5 * torch.sum(t**2 + h**2)
        norm += 1.5 * torch.sum(h**2 * r**2 + t**2 * r**2)

        return self.weight * norm / h.shape[0]

class DURA_ComplexH(Regularizer):
    def __init__(self, weight: float):
        super(DURA_ComplexH, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
       
        h, r, t = factors

        norm += 0.5 * torch.sum(t**2 + h**2)
        res1 = givens_rotations_stretch(r, h)
        res2 = givens_rotations_stretch_T(r, t)
        norm += 1.5 * torch.sum(res1**2 + res2**2)

        return self.weight * norm / h.shape[0]

    
    
class DURA_DistMultH(Regularizer):
    def __init__(self, weight: float):
        super(DURA_DistMultH, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
       
        h, r, t = factors

        norm += torch.sum(t**2 + h**2)
        norm += torch.sum(h**2 * r**2 + t**2 * r**2)

        return self.weight * norm / h.shape[0]
    
class DURA_DistMultH_P(Regularizer):
    def __init__(self, weight: float, w):
        super(DURA_DistMultH_P, self).__init__()
        self.weight = weight
        self.w = w 

    def forward(self, factors):
        norm = 0
       
        h, r, t = factors

        norm += self.w * torch.sum(t**2 + h**2)
        norm += torch.sum(h**2 * r**2 + t**2 * r**2)

        return self.weight * norm / h.shape[0]
    
    
class DURA_RESCAL(Regularizer):
    def __init__(self, weight: float):
        super(DURA_RESCAL, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        
        h, r, t = factors
        norm += torch.sum(h ** 2 + t ** 2)
        norm += torch.sum(
        torch.bmm(r, h.unsqueeze(-1)) ** 2 + torch.bmm(r.transpose(1, 2), t.unsqueeze(-1)) ** 2)
        
        return self.weight * norm / h.shape[0]
    
class DURA_RESCAL_1(Regularizer):
    def __init__(self, weight: float):
        super(DURA_RESCAL_1, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        
        h, r, t = factors
        norm += 2.0 * torch.sum(h ** 2 + t ** 2)
        norm += torch.sum(
        torch.bmm(r, h.unsqueeze(-1)) ** 2 + torch.bmm(r.transpose(1, 2), t.unsqueeze(-1)) ** 2)
        
        return self.weight * norm / h.shape[0]

class DURA_RESCAL_2(Regularizer):
    def __init__(self, weight: float):
        super(DURA_RESCAL_2, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        
        h, r, t = factors
        norm += 0.2 * torch.sum(h ** 2 + t ** 2)
        norm += torch.sum(
        torch.bmm(r, h.unsqueeze(-1)) ** 2 + torch.bmm(r.transpose(1, 2), t.unsqueeze(-1)) ** 2)
        
        return self.weight * norm / h.shape[0]
    
class DURA_RESCAL_H(Regularizer):
    def __init__(self, weight: float):
        super(DURA_RESCAL_H, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        
        h, r, t = factors

        norm += torch.sum(expmap0(h,c) ** 2 + expmap0(t,c) ** 2)
        norm += torch.sum(
        expmap0(torch.bmm(r, h.unsqueeze(-1)), c) ** 2 + expmap0(torch.bmm(r.transpose(1, 2), t.unsqueeze(-1)), c) ** 2)
        
        return self.weight * norm / h.shape[0]

    
class DURA_RESCAL_W(Regularizer):
    def __init__(self, weight: float):
        super(DURA_RESCAL_W, self).__init__()
        self.weight = weight

    def forward(self, factors):
        
        norm = 0
        
        h, r, t = factors
        norm += 2.0 * torch.sum(h ** 2 + t ** 2)
        norm += 0.5 * torch.sum(
            torch.bmm(r, h.unsqueeze(-1)) ** 2 + torch.bmm(r.transpose(1, 2), t.unsqueeze(-1)) ** 2)
        
        return self.weight * norm / h.shape[0]

class DURA_RESCAL_P(Regularizer):
    def __init__(self, weight: float, w):
        super(DURA_RESCAL_P, self).__init__()
        self.weight = weight
        self.w = w

    def forward(self, factors):
        
        norm = 0
        
        h, r, t = factors
        norm += self.w * torch.sum(h ** 2 + t ** 2)
        norm += 0.5 * torch.sum(
        torch.bmm(r, h.unsqueeze(-1)) ** 2 + torch.bmm(r.transpose(1, 2), t.unsqueeze(-1)) ** 2)
        
        return self.weight * norm / h.shape[0]
        
        
        
class Or_Regu(Regularizer):
    def __init__(self, weight: float):
        super(Or_Regu, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
       
        h, r, t = factors
        
        rank = r.shape[1]
        r_reg = torch.bmm(r.transpose(-1,-2), r) - torch.eye(rank, device="cuda").unsqueeze(dim=0)
        norm += torch.sum(r_reg ** 2)

        return self.weight * norm / r.shape[0]
    
class Or_Regu_double(Regularizer):
    def __init__(self, weight: float):
        super(Or_Regu_double, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
       
        h, r, t = factors
        
        rank = r.shape[1]
        r_reg = torch.bmm(r.transpose(-1,-2), r) - torch.eye(rank, device="cuda").unsqueeze(dim=0)
        norm += torch.sum(r_reg ** 2)
        r_reg = torch.bmm(r, r.transpose(-1,-2)) - torch.eye(rank, device="cuda").unsqueeze(dim=0)
        norm += torch.sum(r_reg ** 2)

        return self.weight * norm / r.shape[0]

class Or_Regu_F2(Regularizer):
    def __init__(self, weight: float):
        super(Or_Regu_F2, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
       
        h, r, t = factors
        
        rank = r.shape[1]
        r_reg = torch.bmm(r.transpose(-1,-2), r) - torch.eye(rank, device="cuda").unsqueeze(dim=0)
        norm += 0.1 * torch.sum(r_reg ** 2)
        norm += torch.sum(h ** 2)
        norm += torch.sum(t ** 2)
        return self.weight * norm / r.shape[0]