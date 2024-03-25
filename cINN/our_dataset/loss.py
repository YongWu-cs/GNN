import torch
import torch.nn.functional as F

def NLL_loss(z,log_j,ndim_total):
    return torch.mean(z**2) / 2 - torch.mean(log_j) / ndim_total

def MSE_loss(img1,img2):
    return F.mse_loss(img1, img2, reduction='mean')