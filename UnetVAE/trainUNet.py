from UNetVAE import *
from getDataset import CustomDataset
import torch.optim as optim
import torchvision
from tqdm import tqdm
from setting import *

dataset = CustomDataset(color_dataset_folder='Dataset/Train/ColorImgs', gray_dataset_folder='Dataset/Train/GrayImgs', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model=UNet().to(device)

num_epochs=10
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

total_loss=0.0
'''
for data in tqdm(dataloader):
    Color_Imgs=data[:,0:3,:,:].to(device)
    Gray_Imgs=data[:,3:4,:,:].to(device)
    y=model(Gray_Imgs)
    loss = criterion(y, Gray_Imgs)
    total_loss += loss.item()
'''
best_loss = total_loss / len(dataloader)
best_loss=100
losses = [best_loss]
for i in tqdm(range(num_epochs)):
    total_loss = 0.0
    for data in tqdm(dataloader):
        Color_Imgs=data[:,0:3,:,:].to(device)
        Gray_Imgs=data[:,3:4,:,:].to(device)
        y=model(Gray_Imgs)
        generated_color_imgs=model.generate(Gray_Imgs)
        loss = criterion(y, Gray_Imgs)+criterion(generated_color_imgs, Color_Imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        ind=1  
        test_img=y[ind,:,:,:]
        gray_img=Gray_Imgs[ind,:,:,:]
        generated_color_img=generated_color_imgs[ind,:,:,:]
        torchvision.utils.save_image(test_img, 'test_img.png')
        torchvision.utils.save_image(gray_img, 'gray_img.png')
        torchvision.utils.save_image(generated_color_img, 'generated_color_img.png')
    
    average_loss = total_loss / len(dataloader)
    losses.append(average_loss)
    if average_loss<best_loss:
        best_loss=average_loss
        model.save(path='bestUNet.pth')
    tqdm.write(f'Epoch {i + 1}/{num_epochs}, Loss: {average_loss:.6f}')


with open('Unet_loss.txt', 'w') as f:
    for i, loss in enumerate(losses):
        f.write(f'Epoch {i}/{num_epochs}, Loss: {loss:.6f}\n')

model.save()