import torch 
from torch import nn

# import cross_entropy_loss
import torch.nn.functional as F
import numpy as np
import src.data_augmentation as DA
    

class AdvWmLoss(nn.Module):
    def __init__(self, length_target=63.0, angle_target=1.0):
        super(AdvWmLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.wl = 0.1
        self.wa = 200 
        self.len_t = length_target
        self.agl_t = angle_target

    
    def forward(self, inputs, targets):
        loss_a = torch.log10((1.0-self.cos(inputs, targets))**2)
        loss_l = (torch.norm(inputs, dim=1, keepdim=True)**2-63.0)**2 
        # 62.49 for p-value 0.001; 46.19 for p-value = 0.05

        loss = torch.tensor(-1.0)*torch.mean(self.wl*loss_l + self.wa*loss_a)

        return loss
    

class OnlyLengthLoss(nn.Module):
    def __init__(self):
        super(OnlyLengthLoss, self).__init__()

    def forward(self, inputs, targets):
        outputs = torch.norm(inputs, dim=1, keepdim=True)**2

        loss = torch.mean(outputs)
        return loss
    

class OnlyAngleLoss(nn.Module):
    def __init__(self):
        super(OnlyAngleLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, inputs, targets):
        outputs = self.cos(inputs, targets)
        angle = outputs.clamp(-1, 1)
        loss = torch.mean(angle)

        return loss


class MultiWassersteinLoss(nn.Module):
    # a more stable version of MultiWassersteinloss compared to compute the eigenvalue and eigenvector
    # using cholesky decomposition to compute the square root of the covariance matrix
    # cholesky decomposition requires the covariance matrix to be positive-definite
    # so we add a small constant to the diagonal to ensure positive-definiteness
    def __init__(self):
        super(MultiWassersteinLoss, self).__init__()

    def forward(self, inputs):
        # t_mean [1, features]
        cov = torch.cov(inputs.T)  # the inputs is Batch * features, so we need to transpose it for torch.cov, because the rows are variables and the columns are oberservations in cov
        mean = torch.mean(inputs, dim=0, keepdim=True)
        
        eps = torch.tensor(1e-8)*torch.eye(cov.size(0)).to(cov.device)
        cov = cov + eps  # add a small constant to the diagonal to ensure positive-definiteness

        cov_root = torch.linalg.cholesky(cov)
        wasserstein_loss = torch.norm(mean, dim=1, keepdim=True)**2 + torch.trace(cov - 2*cov_root)+ cov.size(0)

        return wasserstein_loss.squeeze()

class VarianceLoss(nn.Module):
    def __init__(self, mode='L1'):
        super(VarianceLoss, self).__init__()
        self.mode = mode  # 'L1', 'L2', or 'ReLU' 

    def forward(self, inputs):
        cov = torch.cov(inputs.T)
        variances = torch.diag(cov)
        deviations = variances - 1  # since we want the variances to be close to 1 (identity matrix diagonal)
        
        if self.mode == 'L1':
            loss = torch.sum(torch.abs(deviations))
        elif self.mode == 'L2':
            loss = torch.sum(deviations**2)
        elif self.mode == 'ReLU':
            loss = torch.sum(torch.nn.functional.relu(deviations))
        else:
            raise ValueError("Invalid mode. Choose from 'L1', 'L2', or 'ReLU'.")
            
        return loss



