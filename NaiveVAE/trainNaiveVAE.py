from NaiveVAE import *
from getDataset import CustomDataset
import torch.optim as optim
import torchvision
from tqdm import tqdm
from setting import *

dataset = CustomDataset(color_dataset_folder='Dataset/Train/ColorImgs', gray_dataset_folder='Dataset/Train/GrayImgs', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model=NaiveVAE(latent_dim=latent_dim,act_func=act_func).to(device)
model.load(path='bestNaiveVAE.pth')

w=0.0003
num_epochs=10
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

total_loss=0.0
total_loss1=0.0
total_loss2=0.0
'''
for data in tqdm(dataloader):
    Color_Imgs,Gray_Imgs=data
    Color_Imgs=Color_Imgs.to(device)
    Gray_Imgs=Gray_Imgs.to(device)
    mu,logvar,z=model(Color_Imgs,Gray_Imgs)
    loss1 = criterion(z, Color_Imgs)
    loss2=-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    loss=loss1+w*loss2
    total_loss += loss.item()
    total_loss1 += loss1.item()
    total_loss2 += loss2.item()

    
best_loss = total_loss / len(dataloader)
best_loss1 = total_loss1 / len(dataloader)
best_loss2 = total_loss2 / len(dataloader)
'''
best_loss = 100
best_loss1 = 100
best_loss2 = 100
losses = [best_loss]
losses1 = [best_loss1]
losses2 = [best_loss2]
for i in tqdm(range(num_epochs)):
    total_loss = 0.0
    total_loss1 = 0.0
    total_loss2 = 0.0
    for data in tqdm(dataloader):
        Color_Imgs,Gray_Imgs=data
        Color_Imgs=Color_Imgs.to(device)
        Gray_Imgs=Gray_Imgs.to(device)
        mu,logvar,z=model(Color_Imgs,Gray_Imgs)
        loss1 = criterion(z, Color_Imgs)
        loss2=-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        #w=torch.exp(-loss1)
        loss=loss1+w*loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_loss1 += loss1.item()
        total_loss2 += loss2.item()

        ind=1   
        original_img=Color_Imgs[ind,:,:,:]
        test_img=z[ind,:,:,:]
        gray_img=Gray_Imgs[ind,:,:,:]
        z1=torch.ones((Gray_Imgs.size(0),latent_dim))*5
        z2=torch.ones((Gray_Imgs.size(0),latent_dim))*10
        r_x=model.generate(Gray_Imgs,device,z1)[ind,:,:,:]
        r_x1=model.generate(Gray_Imgs,device,z2)[ind,:,:,:]
        torchvision.utils.save_image(test_img, 'test_img.png')
        torchvision.utils.save_image(original_img, 'original_img.png')
        torchvision.utils.save_image(gray_img, 'gray_img.png')
        torchvision.utils.save_image(r_x, 'generated_img1.png')
        torchvision.utils.save_image(r_x1, 'generated_img2.png')

    average_loss = total_loss / len(dataloader)
    losses.append(average_loss)
    average_loss1 = total_loss1 / len(dataloader)
    losses1.append(average_loss1)
    average_loss2 = total_loss2 / len(dataloader)
    losses2.append(average_loss2)
    if average_loss<best_loss:
        best_loss=average_loss
        model.save(path='bestNaiveVAE.pth')
    tqdm.write(f'Epoch {i + 1}/{num_epochs}, Loss: {average_loss:.6f}, Loss1: {average_loss1:.6f}, Loss2: {average_loss2:.6f}')


with open('NaiveVAE_loss.txt', 'w') as f:
    # 将损失值写入文本文件
    for i, loss in enumerate(losses):
        f.write(f'Epoch {i + 1}/{num_epochs}, Loss: {loss:.6f}\n')

with open('NaiveVAE_loss1.txt', 'w') as f:
    # 将损失值写入文本文件
    for i, loss in enumerate(losses):
        f.write(f'Epoch {i + 1}/{num_epochs}, Loss: {loss1:.6f}\n')

with open('NaiveVAE_loss2.txt', 'w') as f:
    # 将损失值写入文本文件
    for i, loss in enumerate(losses):
        f.write(f'Epoch {i + 1}/{num_epochs}, Loss: {loss2:.6f}\n')

model.save()