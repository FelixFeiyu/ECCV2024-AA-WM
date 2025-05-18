import torch
import torch.nn.functional as F
import numpy as np
from src.utils import set_random_seed

def normalizedobs(observations):
    # this fucntion is to make every row vector of an observation matrix follows the N(0, I) distributions
    batchize = observations.shape[0]
    om = torch.mean(observations, dim=0, keepdim=True).repeat(batchize, 1)
    oc = torch.cov(observations.T)
    eps = torch.tensor(1e-15).to(observations.device)
    eigenvalue, eigenvector = torch.linalg.eig(oc)
    cov_root = torch.matmul(eigenvector, torch.matmul(torch.diag(torch.sqrt(eigenvalue+eps)), eigenvector.T)).real
    observations = torch.mm(observations - om, torch.linalg.pinv(cov_root))
    
    return observations

def naive_signatures(img_vec, seed = 0):

    set_random_seed(seed)
    y_signature = normalizedobs(torch.randn_like(img_vec))

    return y_signature


def alignment_signatures(img_vec, seed = 0):
    set_random_seed(seed)
    y_signature = torch.randn_like(img_vec)
    y_signature = F.normalize(y_signature, p=2, dim=-1)
    cosine_similarities = F.cosine_similarity(img_vec, y_signature, dim=-1, eps=1e-6)
    mask = cosine_similarities < 0
    y_signature[mask] = -y_signature[mask]

    return y_signature

def orthogonal_signatures(img_vec, seed = 0):

    set_random_seed(seed)
    batch_size, dim = img_vec.shape
    a = img_vec.detach().clone()

    b_perp = torch.randn_like(img_vec)
    dot_product = torch.sum(b_perp * a, dim=1, keepdim=True)
    norms_a_squared = torch.norm(a, dim=1, keepdim=True) ** 2

    b_perp -= dot_product * a / norms_a_squared
    norms_b_perp = torch.norm(b_perp, dim=1, keepdim=True)
    
    # Check if any vector is too close to zero and regenerate
    while torch.any(norms_b_perp < 1e-10):
        mask = norms_b_perp < 1e-10
        b_perp[mask] = torch.randn(mask.sum(), dim)
        dot_product = torch.sum(b_perp * a, dim=1, keepdim=True)
        b_perp -= dot_product * a / norms_a_squared
        norms_b_perp = torch.norm(b_perp, dim=1, keepdim=True)

    b_perp_unit_batch = b_perp / norms_b_perp
    b_final = b_perp_unit_batch

    return  b_final