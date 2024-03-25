from torchvision import transforms
import numpy as np
import warnings
import torch
import torch.nn as nn
import random
width_height=128
batch_size=32
latent_dim=512*4*4

class Swish(torch.nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * nn.Sigmoid(x)
    
act_func=nn.ReLU()

transform = transforms.Compose([
        transforms.Resize((width_height, width_height)),
        transforms.ToTensor()
    ])


warnings.filterwarnings("ignore", category=UserWarning)

seed = 1024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")