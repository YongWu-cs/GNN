import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        #encoder
        self.enlayer1=nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            )
        self.enlayer2=nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.enlayer3=nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.enlayer4=nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.enlayer5=nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        )
        self.enlayer6=nn.Flatten()

        # decoder

        self.delayer6=nn.Unflatten(1, (512, 4, 4))
        self.delayer5=nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2)
        )
        self.delayer4=nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2)
        )
        self.delayer3=nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2)
        )
        self.delayer2=nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True) 
        )
        self.delayer1=nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True), 
        )

    def save(self, path = "UNet.pth"):
        torch.save(self.state_dict(), path)
    
    def rgb2gray(self,rgb_imgs):
        gray_imgs=0.299*rgb_imgs[:,0:1,:,:]+0.587*rgb_imgs[:,1:2,:,:]+0.114*rgb_imgs[:,2:3,:,:]
        #gray_imgs=1/3*rgb_imgs[:,0:1,:,:]+1/3*rgb_imgs[:,1:2,:,:]+1/3*rgb_imgs[:,2:3,:,:]
        return gray_imgs
    
    
    def load(self, path = "UNet.pth"):
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

    def forward(self, x):

        #encoder
        e1=self.enlayer1(x)
        e2=self.enlayer2(e1)
        e3=self.enlayer3(e2)
        e4=self.enlayer4(e3)
        e5=self.enlayer5(e4)
        z=self.enlayer6(e5)

        #decoder
        d5=self.delayer6(z)
        d4=self.delayer5(d5)
        d4=torch.cat((e3, d4), dim=1)
        d3=self.delayer4(d4)
        d3=torch.cat((e2, d3), dim=1)
        d2=self.delayer3(d3)
        d2=torch.cat((e1, d2), dim=1)
        d1=self.delayer2(d2)
        y=self.rgb2gray(d1)
        print(e1.shape)
        print(e2.shape)
        print(e3.shape)
        print(e4.shape)
        print(e5.shape)
        print(z.shape)
        print(d5.shape)
        print(d4.shape)
        print(d3.shape)
        print(d2.shape)
        print(d1.shape)
        print(y.shape)
        return y
    
    def generate(self, x, t=None):

        #encoder
        e1=self.enlayer1(x)
        e2=self.enlayer2(e1)
        e3=self.enlayer3(e2)
        e4=self.enlayer4(e3)
        e5=self.enlayer5(e4)
        z=self.enlayer6(e5)

        if t!=None:
            z=z+t

        #decoder
        d5=self.delayer6(z)
        d4=self.delayer5(d5)
        d4=torch.cat((e3, d4), dim=1)
        d3=self.delayer4(d4)
        d3=torch.cat((e2, d3), dim=1)
        d2=self.delayer3(d3)
        d2=torch.cat((e1, d2), dim=1)
        d1=self.delayer2(d2)
        return d1

    def freezeenlayer(self):
        for name, param in self.named_parameters():
            if "enlayer" in name:  # 根据层的名称来判断是否冻结
                param.requires_grad = False
    
    def freezedelayer(self):
        for name, param in self.named_parameters():
            if "delayer" in name:  # 根据层的名称来判断是否冻结
                param.requires_grad = False
    
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
    
class UNetVAE(nn.Module):
    def __init__(self,latent_dim=20,act_func=nn.ReLU()):
        super(UNetVAE, self).__init__()
        self.latent_dim=latent_dim
        self.unet=UNet()
        self.encoder=VAEEncoder(self.latent_dim,act_func)
        self.linear=nn.Sequential(
            nn.Linear(self.latent_dim,512*4*4),
            act_func,

        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def save(self, path = "UNetVAE.pth"):
        torch.save(self.state_dict(), path)
    
    
    def load(self, path = "UNetVAE.pth"):
        try:
            self.load_state_dict(torch.load(path))
            print('Model loading success')
            return True
        except:
            print('Model loading failed')
            return False
        
    def rgb2gray(self,rgb_imgs):
        gray_imgs=0.299*rgb_imgs[:,0:1,:,:]+0.587*rgb_imgs[:,1:2,:,:]+0.114*rgb_imgs[:,2:3,:,:]
        return gray_imgs  
      
    def forward(self,color_imgs,gray_imgs):
        mu, logvar = self.encoder(color_imgs)
        z=self.reparameterize(mu,logvar)
        t=z
        generated_color_imgs=self.unet.generate(gray_imgs,t)
        generated_gray_imgs=self.rgb2gray(generated_color_imgs)
        return mu, logvar, generated_color_imgs, generated_gray_imgs

    def generate(self,gray_imgs,device,z=None):
        if z==None:
            z=torch.randn(gray_imgs.size(0), self.latent_dim).to(device)
            
        else:
            z=z.to(device)
        
        t=z
        generated_color_imgs=self.unet.generate(gray_imgs,t)
        return generated_color_imgs