import torch
import torch.nn.functional as F

def variation_loss(img,ndim_total, weight=1e-3):
    batch_size = img.size(0)
    h_variation = torch.sum(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    w_variation = torch.sum(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    tv_loss = weight * (h_variation + w_variation) / (batch_size*ndim_total)
    return tv_loss

def NLL_loss(z,log_j,ndim_total):
    return torch.mean(z**2) / 2 - torch.mean(log_j) / ndim_total

def MSE_loss(img1,img2):
    return F.mse_loss(img1, img2, reduction='mean')