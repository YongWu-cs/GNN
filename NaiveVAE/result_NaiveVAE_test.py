import torchvision
from getDataset import CustomDataset
from NaiveVAE import *
from tqdm import tqdm
from setting import *

dataset = CustomDataset(color_dataset_folder='Dataset/Test/ColorImgs', gray_dataset_folder='Dataset/Test/GrayImgs', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model=NaiveVAE(latent_dim=latent_dim,act_func=act_func).to(device)
model.load(path='bestNaiveVAE.pth')
model.eval()

for data in tqdm(dataloader):
    Color_Imgs,Gray_Imgs=data
    Color_Imgs=Color_Imgs.to(device)
    Gray_Imgs=Gray_Imgs.to(device)
    z=model.generate(Gray_Imgs,device)

    ind=5
    original_img=Color_Imgs[ind,:,:,:]
    gray_img=Gray_Imgs[ind,:,:,:]
    result_img=z[ind,:,:,:]
    torchvision.utils.save_image(original_img, 'color_img.png')
    torchvision.utils.save_image(gray_img, 'gray_img.png')
    torchvision.utils.save_image(result_img, 'result_img.png')
    break