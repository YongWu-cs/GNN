import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch

class VAEEncoder(nn.Module):
    def __init__(self,latent_dim=20,act_func=nn.ReLU()):
        super(VAEEncoder, self).__init__()
        self.conv2=nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm2d(16),
            act_func,
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm2d(32),
            act_func,
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm2d(64),
            act_func,
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm2d(128),
            act_func,
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm2d(256),
            act_func,
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(256*4*4, latent_dim)
        self.fc_logvar = nn.Linear(256*4*4, latent_dim)
    
    def forward(self, x):
        x = self.conv2(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
class VAEDecoder(nn.Module):
    def __init__(self,latent_dim=20,act_func=nn.ReLU()):
        super(VAEDecoder, self).__init__()

        self.layer2=nn.Sequential(
            nn.Flatten()
        )
        self.conv2=nn.Sequential(
            nn.Linear(512 * 4 * 4 , 128 * 16 * 16),
            act_func,
            nn.Unflatten(1, (128, 16, 16)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            #nn.BatchNorm2d(64),
            act_func,
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            #nn.BatchNorm2d(32),
            act_func,
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    
    def forward(self, z, features):
        features=self.layer2(features)
        r_x=self.conv2(features+z)
        return r_x

class NaiveVAE(nn.Module):
    def __init__(self,latent_dim=20,act_func=nn.ReLU()):
        super(NaiveVAE, self).__init__()
        self.latent_dim=latent_dim
        self.vaeEncoder=VAEEncoder(self.latent_dim,act_func)
        self.vaeDecoder=VAEDecoder(self.latent_dim,act_func)
        self.vggmodel=models.vgg19(pretrained=True).features
        self.vggmodel.eval()

    def save(self, path = "NaiveVAE.pth"):
        torch.save(self.state_dict(), path)

    def load(self, path = "NaiveVAE.pth"):
        try:
            self.load_state_dict(torch.load(path))
            print('Model loading success')
            return True
        except:
            print('Model loading failed')
            return False
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, color_imgs,gray_imgs):
        features=self.vggmodel(gray_imgs)
        mu, logvar = self.vaeEncoder(color_imgs)
        z=self.reparameterize(mu,logvar)
        r_x=self.vaeDecoder(z,features)
        return mu,logvar,r_x
    
    def generate(self,gray_imgs,device,z=None):
        
        features=self.vggmodel(gray_imgs)
        if z==None:
            z=torch.randn(gray_imgs.size(0), self.latent_dim).to(device)
        else:
            z=z.to(device)
        r_x=self.vaeDecoder(z,features)
        return r_x