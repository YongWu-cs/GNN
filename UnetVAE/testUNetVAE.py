from getDataset import CustomDataset
from UNetVAE import *
from tqdm import tqdm
from setting import *
import torchvision

dataset = CustomDataset(color_dataset_folder='Dataset/Train/ColorImgs', gray_dataset_folder='Dataset/Train/GrayImgs', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model=UNetVAE(latent_dim=latent_dim,act_func=act_func).to(device)
model.load(path='bestUNetVAE.pth')
#model.unet.load('bestUNet.pth')
model.eval()

for data in tqdm(dataloader):
    Color_Imgs=data[:,0:3,:,:].to(device)
    Gray_Imgs=data[:,3:4,:,:].to(device)
    z=torch.ones((Gray_Imgs.size(0),latent_dim))*0
    y=model.generate(Gray_Imgs,device,z)

    ind=3
    color_img=Color_Imgs[ind,:,:,:]
    gray_img=Gray_Imgs[ind,:,:,:]
    result_img=y[ind,:,:,:]
    torchvision.utils.save_image(color_img, 'color_img.png')
    torchvision.utils.save_image(gray_img, 'gray_img.png')
    torchvision.utils.save_image(result_img, 'test_img_1.png')
    break