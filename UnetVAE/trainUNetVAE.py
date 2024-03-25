from UNetVAE import *
from getDataset import CustomDataset
import torch.optim as optim
import torchvision
from tqdm import tqdm
from setting import *

dataset = CustomDataset(color_dataset_folder='Dataset/Train/ColorImgs', gray_dataset_folder='Dataset/Train/GrayImgs', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model=UNetVAE(latent_dim=latent_dim,act_func=act_func).to(device)
'''
model.load(path='bestUNetVAE.pth')
'''
model.unet.load(path='bestUNet.pth')
model.unet.freezeenlayer()

w1=0.00003
num_epochs=20
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.000001)

total_loss=0.0
total_loss1=0.0
total_loss2=0.0
total_loss3=0.0

for data in tqdm(dataloader):
    Color_Imgs=data[:,0:3,:,:].to(device)
    Gray_Imgs=data[:,3:4,:,:].to(device)
    mu, logvar, generated_color_imgs, generated_gray_imgs=model(Color_Imgs,Gray_Imgs)

    loss1 = criterion(generated_color_imgs, Color_Imgs)
    loss2 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss3 = criterion(generated_gray_imgs, Gray_Imgs)
    loss=loss1+w1*loss2
    total_loss1 += loss1.item()
    total_loss2 += loss2.item()
    total_loss3 += loss3.item()
    total_loss += loss.item()

best_loss = total_loss / len(dataloader)
best_loss1 = total_loss1 / len(dataloader)
best_loss2 = total_loss2 / len(dataloader)
best_loss3 = total_loss3 / len(dataloader)

losses = [best_loss]
losses1 = [best_loss1]
losses2 = [best_loss2]
losses3 = [best_loss3]
for i in tqdm(range(num_epochs)):
    total_loss = 0.0
    total_loss1 = 0.0
    total_loss2 = 0.0
    total_loss3 = 0.0
    for data in tqdm(dataloader):
        Color_Imgs=data[:,0:3,:,:].to(device)
        Gray_Imgs=data[:,3:4,:,:].to(device)
        mu, logvar, generated_color_imgs, generated_gray_imgs=model(Color_Imgs,Gray_Imgs)

        loss1 = criterion(generated_color_imgs, Color_Imgs)
        loss2 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss3 = criterion(generated_gray_imgs, Gray_Imgs)
        loss=loss1+w1*loss2
        total_loss1 += loss1.item()
        total_loss2 += loss2.item()
        total_loss3 += loss3.item()
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ind=1
        color_img=Color_Imgs[ind,:,:,:]
        gray_img=Gray_Imgs[ind,:,:,:]  
        generated_color_img=generated_color_imgs[ind,:,:,:]
        generated_gray_img=generated_gray_imgs[ind,:,:,:]
        torchvision.utils.save_image(color_img, 'color_img.png')
        torchvision.utils.save_image(gray_img, 'gray_img.png')
        torchvision.utils.save_image(generated_color_img, 'generated_color_img.png')
        torchvision.utils.save_image(generated_gray_img, 'generated_gray_img.png')
    
    average_loss = total_loss / len(dataloader)
    average_loss1 = total_loss1 / len(dataloader)
    average_loss2 = total_loss2 / len(dataloader)
    average_loss3 = total_loss3 / len(dataloader)
    losses.append(average_loss)
    losses1.append(average_loss1)
    losses2.append(average_loss2)
    losses3.append(average_loss3)

    if average_loss<best_loss:
        best_loss=average_loss
        model.save(path='bestUNetVAE.pth')
    tqdm.write(f'Epoch {i + 1}/{num_epochs}, Loss: {average_loss:.6f}, Loss1: {average_loss1:.6f}, Loss2: {average_loss2:.6f}, Loss3: {average_loss3:.6f}')


with open('UNetVAE_loss.txt', 'w') as f:
    for i, loss in enumerate(losses):
        f.write(f'Epoch {i}/{num_epochs}, Loss: {loss:.6f}\n')
with open('UNetVAE_loss1.txt', 'w') as f:
    for i, loss in enumerate(losses1):
        f.write(f'Epoch {i}/{num_epochs}, Loss: {loss:.6f}\n')
with open('UNetVAE_loss2.txt', 'w') as f:
    for i, loss in enumerate(losses2):
        f.write(f'Epoch {i}/{num_epochs}, Loss: {loss:.6f}\n')        
with open('UNetVAE_loss3.txt', 'w') as f:
    for i, loss in enumerate(losses3):
        f.write(f'Epoch {i}/{num_epochs}, Loss: {loss:.6f}\n')

model.save()