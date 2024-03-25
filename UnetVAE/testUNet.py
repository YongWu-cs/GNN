import torchvision
from getDataset import CustomDataset
from UNetVAE import *
from tqdm import tqdm
from setting import *

dataset = CustomDataset(color_dataset_folder='Dataset/Test/ColorImgs', gray_dataset_folder='Dataset/Test/GrayImgs', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model=UNet().to(device)
model.load(path='bestUNet.pth')
model.eval()

for data in tqdm(dataloader):
    Color_Imgs=data[:,0:3,:,:].to(device)
    Gray_Imgs=data[:,3:4,:,:].to(device)
    y=model(Gray_Imgs)

    ind=31
    original_img=Gray_Imgs[ind,:,:,:]
    result_img=y[ind,:,:,:]
    torchvision.utils.save_image(original_img, 'gray_img.png')
    torchvision.utils.save_image(result_img, 'test_img.png')
    break
