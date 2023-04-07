"""Euclidean operations utils functions."""

import torch


def euc_sqdistance(x, y, eval_mode=False):
    """Compute euclidean squared distance between tensors.

    Args:
        x: torch.Tensor of shape (N1 x d)
        y: torch.Tensor of shape (N2 x d)
        eval_mode: boolean

    Returns:
        torch.Tensor of shape N1 x 1 with pairwise squared distances if eval_mode is false
        else torch.Tensor of shape N1 x N2 with all-pairs distances

    """
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    if eval_mode:
        y2 = y2.t()
        xy = x @ y.t()
    else:
        assert x.shape[0] == y.shape[0]
        xy = torch.sum(x * y, dim=-1, keepdim=True)
    return x2 + y2 - 2 * xy

def euc_sqdistance_Ma(x, y, u, eval_mode=False):
    """Compute euclidean squared distance between tensors.

    Args:
        u: torch.Tensor of shape (1 x d)
        x: torch.Tensor of shape (N1 x d)
        y: torch.Tensor of shape (N2 x d)
        eval_mode: boolean

    Returns:
        torch.Tensor of shape N1 x 1 with pairwise squared distances if eval_mode is false
        else torch.Tensor of shape N1 x N2 with all-pairs distances

    """
    x2 = torch.sum(u * x * x, dim=-1, keepdim=True)
    y2 = torch.sum(u * y * y, dim=-1, keepdim=True)
    if eval_mode:
        y2 = y2.t()
        xy = (u*x) @ y.t()
    else:
        assert x.shape[0] == y.shape[0]
        xy = torch.sum(u * x * y, dim=-1, keepdim=True)
    return x2 + y2 - 2 * xy

def euc_sqdistance_SL(x, y, eval_mode=False):
    """Compute euclidean squared distance between tensors.

    Args:
        u: torch.Tensor of shape (1 x d)
        x: torch.Tensor of shape (N1 x d)
        y: torch.Tensor of shape (N2 x d)
        eval_mode: boolean

    Returns:
        torch.Tensor of shape N1 x 1 with pairwise squared distances if eval_mode is false
        else torch.Tensor of shape N1 x N2 with all-pairs distances

    """
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    if eval_mode:
        y2 = y2.t()
        xy = x @ y.t()
    else:
        assert x.shape[0] == y.shape[0]
        xy = torch.sum( x * y, dim=-1, keepdim=True)
    return 2 * x2*y2 - 2 * xy

def euc_sqdistance_Du(x, y, eval_mode=False):
    """Compute euclidean squared distance between tensors.

    Args:
        x: torch.Tensor of shape (N1 x d)
        y: torch.Tensor of shape (N1 x N2 x d)
        eval_mode: boolean

    Returns:
        torch.Tensor of shape N1 x 1 with pairwise squared distances if eval_mode is false
        else torch.Tensor of shape N1 x N2 with all-pairs distances

    """
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=False)    # (N1, N2)
    xy = torch.matmul(y, x.unsqueeze(dim=-1).squeeze())  # (N1, N2)
                      
    return x2 + y2 - 2 * xy
    

def givens_rotations(r, x):
    """Givens rotations.

    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to rotate

    Returns:
        torch.Tensor os shape (N x d) representing rotation of x by r
    """
    givens = r.view((r.shape[0], -1, 2))
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    x = x.view((r.shape[0], -1, 2))
    x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_rot.view((r.shape[0], -1))

def givens_rotations_stretch(r, x):
    """Givens rotations.

    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to rotate

    Returns:
        torch.Tensor os shape (N x d) representing rotation of x by r
    """
    givens = r.view((r.shape[0], -1, 2))
    x = x.view((r.shape[0], -1, 2))
    x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_rot.view((r.shape[0], -1))

def givens_rotations_stretch_T(r, x):
    """Givens rotations.

    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to rotate

    Returns:
        torch.Tensor os shape (N x d) representing rotation of x by r
    """
    givens = r.view((r.shape[0], -1, 2))
    x = x.view((r.shape[0], -1, 2))
    x_rot = givens[:, :, 0:1] * x - givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_rot.view((r.shape[0], -1))

def givens_reflection(r, x):
    """Givens reflections.

    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to reflect

    Returns:
        torch.Tensor os shape (N x d) representing reflection of x by r
    """
    givens = r.view((r.shape[0], -1, 2))
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    x = x.view((r.shape[0], -1, 2))
    x_ref = givens[:, :, 0:1] * torch.cat((x[:, :, 0:1], -x[:, :, 1:]), dim=-1) + givens[:, :, 1:] * torch.cat(
        (x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_ref.view((r.shape[0], -1))


def get_rot_rel(rel, num_rot, rank):
    '''
    get the orthogonal matrix
    rel: (num_rel x (rank * num_rot))
    return (num_rel x rank x rank)
    '''
    assert rank % num_rot == 0
    batch_size = rel.shape[0]
    rel = rel.view(-1, num_rot, num_rot)
    q, _ = torch.qr(rel)   # q (Bd x num_rot x num_rot)
    q_chunk = torch.chunk(q, batch_size, dim=0)
    q_block = [torch.block_diag(*(tmp).squeeze()) for tmp in q_chunk]
    ans = torch.stack(q_block, dim=0)

    return ans


def _onorm(r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8):
    denominator_0 = r_1 ** 2 + r_2 ** 2 + r_3 ** 2 + r_4 ** 2
    denominator_1 = torch.sqrt(denominator_0)
    deno_cross = r_5 * r_1 + r_6 * r_2 + r_7 * r_3 + r_8 * r_4

    r_5 = r_5 - deno_cross / denominator_0 * r_1
    r_6 = r_6 - deno_cross / denominator_0 * r_2
    r_7 = r_7 - deno_cross / denominator_0 * r_3
    r_8 = r_8 - deno_cross / denominator_0 * r_4

    r_1 = r_1 / denominator_1
    r_2 = r_2 / denominator_1
    r_3 = r_3 / denominator_1
    r_4 = r_4 / denominator_1
    
    return r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8

def _omult(a_0, a_1, a_2, a_3, b_0, b_1, b_2, b_3, c_0, c_1, c_2, c_3, d_0, d_1, d_2, d_3):

    h_0=a_0*c_0-a_1*c_1-a_2*c_2-a_3*c_3
    h1_0=a_0*d_0+b_0*c_0-a_1*d_1-b_1*c_1-a_2*d_2-b_2*c_2-a_3*d_3-b_3*c_3
    h_1=a_0*c_1+a_1*c_0+a_2*c_3-a_3*c_2
    h1_1=a_0*d_1+b_0*c_1+a_1*d_0+b_1*c_0+a_2*d_3+b_2*c_3-a_3*d_2-b_3*c_2
    h_2=a_0*c_2-a_1*c_3+a_2*c_0+a_3*c_1
    h1_2=a_0*d_2+b_0*c_2-a_1*d_3-b_1*c_3+a_2*d_0+b_2*c_0+a_3*d_1+b_3*c_1
    h_3=a_0*c_3+a_1*c_2-a_2*c_1+a_3*c_0
    h1_3=a_0*d_3+b_0*c_3+a_1*d_2+b_1*c_2-a_2*d_1-b_2*c_1+a_3*d_0+b_3*c_0

    return  (h_0,h_1,h_2,h_3,h1_0,h1_1,h1_2,h1_3)

def _omult_T(a_0, a_1, a_2, a_3, b_0, b_1, b_2, b_3, c_0, c_1, c_2, c_3, d_0, d_1, d_2, d_3):

    h_0=a_0*c_0+a_1*c_1+a_2*c_2+a_3*c_3+b_0*d_0+b_1*d_1+b_2*d_2+b_3*d_3
    h_1=-a_0*c_1+a_1*c_0-a_2*c_3+a_3*c_2-b_0*d_1+b_1*d_0-b_2*d_3+b_3*d_2
    h_2=-a_0*c_2+a_1*c_3+a_2*c_0-a_3*c_1-b_0*d_2+b_1*d_3+b_2*d_0-b_3*d_1
    h_3=-a_0*c_3-a_1*c_2+a_2*c_1+a_3*c_0-b_0*d_3-b_1*d_2+b_2*d_1+b_3*d_0
    h1_0 = b_0*c_0+b_1*c_1+b_2*c_2+b_3*c_3
    h1_1 = -b_0*c_1+b_1*c_0-b_2*c_3+b_3*c_2
    h1_2 = -b_0*c_2+b_1*c_3+b_2*c_0-b_3*c_1
    h1_3 = -b_0*c_3-b_1*c_2+b_2*c_1+b_3*c_0

    return  (h_0,h_1,h_2,h_3,h1_0,h1_1,h1_2,h1_3)

def given_8_rotation(r, x):
    # dual quaternion numerber 
    givens = r.view((r.shape[0], -1, 8))
    r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 = givens[:, :, 0:1], givens[:, :, 1:2], givens[:, :, 2:3], givens[:, :, 3:4], givens[:, :, 4:5], givens[:, :, 5:6], givens[:, :, 6:7], givens[:, :, 7:8]
    r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 = _onorm(r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8)
    x = x.view((r.shape[0], -1, 8))
    x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8 = x[:, :, 0:1], x[:, :, 1:2], x[:, :, 2:3], x[:, :, 3:4], x[:, :, 4:5], x[:, :, 5:6], x[:, :, 6:7], x[:, :, 7:8]
    o_1, o_2, o_3, o_4, o_5, o_6, o_7, o_8 = _omult(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8,
                                                        r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8)
    
    return torch.cat((o_1, o_2, o_3, o_4, o_5, o_6, o_7, o_8), dim=-1).view((r.shape[0], -1))

def given_8_rotation_T(r, x):
    # dual quaternion numerber 
    givens = r.view((r.shape[0], -1, 8))
    r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 = givens[:, :, 0:1], givens[:, :, 1:2], givens[:, :, 2:3], givens[:, :, 3:4], givens[:, :, 4:5], givens[:, :, 5:6], givens[:, :, 6:7], givens[:, :, 7:8]
    r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 = _onorm(r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8)
    x = x.view((r.shape[0], -1, 8))
    x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8 = x[:, :, 0:1], x[:, :, 1:2], x[:, :, 2:3], x[:, :, 3:4], x[:, :, 4:5], x[:, :, 5:6], x[:, :, 6:7], x[:, :, 7:8]
    o_1, o_2, o_3, o_4, o_5, o_6, o_7, o_8 = _omult_T(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8,
                                                        r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8)
    
    return torch.cat((o_1, o_2, o_3, o_4, o_5, o_6, o_7, o_8), dim=-1).view((r.shape[0], -1))