import numpy as np
from skimage import color
from multiprocessing import Pool
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import instance_config as config
import cv2

#Offsets: (23.530156800473875, 1.101879475791363, 1.5505393086563068)
#Scales: (29.745579918550185, 6.657613120043208, 8.61431768136933)
batch_size = config.batch_size
offsets = (45.60540338353164, 2.561262893902535, 6.502379186955269)
scales  = (27.067869578097213, 12.064396872062012, 16.395105370918557)

# offsets = (43.289818075057845, 4.48474443072883, 6.39756368348716)
# scales  = (27.042045552946185, 13.0837885377327, 16.508323262496695)

def apply_jointBilateralFilter(args):
    L, ab_channel, i, j = args
    filtered_channel = cv2.ximgproc.jointBilateralFilter(L, ab_channel, 5, 10, 5)
    return (filtered_channel, i, j)

def parallel_jointBilateralFilter(L, ab):
    assert L.shape[0] == ab.shape[0] and len(ab.shape) == 4, "L and ab must have compatible shapes."

    args_list = []
    for i in range(ab.shape[0]):
        for j in range(ab.shape[3]):
            args_list.append((L[i], ab[i, :, :, j], i, j))
            
    with Pool() as pool:
        results = pool.map(apply_jointBilateralFilter, args_list)
    
    for filtered_channel, i, j in results:
        ab[i, :, :, j] = filtered_channel
    
    return ab

def norm_lab_to_rgb(L, ab, norm=True):
    L = np.array(L).transpose((0, 2, 3, 1)).astype(dtype=np.float32)
    ab = np.array(ab).transpose((0, 2, 3, 1)).astype(dtype=np.float32)
    ab_unsample=np.zeros([ab.shape[0],L.shape[1],L.shape[2],ab.shape[3]])
    for i in range(ab.shape[0]):
        ab_unsample[i]=cv2.resize(ab[i], (L.shape[1],L.shape[2]), interpolation=cv2.INTER_CUBIC)
    ab_unsample=ab_unsample.astype(np.float32)
    ab = parallel_jointBilateralFilter(L, ab_unsample)
    lab=np.concatenate([L,ab],axis=-1)
    lab=lab.transpose((0, 3,1,2))
    for i in range(1 + 2*norm):
        lab[:, i] = lab[:, i] * scales[i] + offsets[i]
    lab[:, 0]=np.clip(lab[:, 0],0.,100.)
    lab[:, 1:]=np.clip(lab[:,1:],-128, 128)
    rgb = [color.lab2rgb(np.transpose(l, (1, 2, 0))).transpose(2, 0, 1) for l in lab]

    if isinstance(rgb, np.ndarray):
        return rgb
    else:
        return np.array(rgb)

class LabColorDataset(Dataset):
    def __init__(self, file_list, transform=None, noise=False,mode="train"):

        self.files = file_list
        self.transform = transform
        self.to_tensor = T.ToTensor()
        self.noise = noise
        self.mode=mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        im = Image.open(self.files[idx])
        if self.transform:
            im = self.transform(im)
        im = self.to_tensor(im).numpy()

        im = np.transpose(im, (1,2,0))
        if im.shape[2] != 3:
            im = np.stack([im[:,:,0]]*3, axis=2)
        im = color.rgb2lab(im).transpose((2, 0, 1))

        for i in range(3):
            im[i] = (im[i] - offsets[i]) / scales[i]
        im = torch.Tensor(im)
        if self.noise:
            im += 0.005 * torch.rand_like(im)
        if self.mode=="test":
            return im,self.files[idx]
        else:
            return im
    
###augment
transf_train = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomResizedCrop((config.image_h_w,config.image_h_w), scale=(0.4, 1.))
])

transf_val_test = T.Compose([
    T.Resize(size=(config.image_h_w,config.image_h_w)),
])

import os
def get_image_path(data_type="train"):
    folder_path = config.dataset_path+"/"+data_type
    res=[]
    for file in os.listdir(folder_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            res.append(os.path.join(folder_path, file))
    return res

if config.mode=="train":
    train_list=get_image_path("train")
    val_list=get_image_path("val")
    train_data = LabColorDataset(train_list, transf_train, noise=True)
    val_data  =  LabColorDataset(val_list,  transf_val_test)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8,  pin_memory=True, drop_last=config.train_drop_last)
    val_loader = DataLoader(val_data,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
else:
    test_list=get_image_path("test")
    test_data  = LabColorDataset(test_list,  transf_val_test,mode="test")
    test_loader = DataLoader(test_data,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

if __name__=="__main__":
    print(len(train_loader))