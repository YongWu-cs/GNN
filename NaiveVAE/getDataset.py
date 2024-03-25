from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, color_dataset_folder, gray_dataset_folder, transform=None):
        self.color_dataset_folder = color_dataset_folder
        self.gray_dataset_folder = gray_dataset_folder
        self.transform = transform
        self.image_filenames = [filename for filename in os.listdir(color_dataset_folder) if filename.endswith('.jpg') or filename.endswith('.png')]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        color_img_name = os.path.join(self.color_dataset_folder, self.image_filenames[idx])
        color_img = Image.open(color_img_name)
        
        gray_img_name = os.path.join(self.gray_dataset_folder, self.image_filenames[idx].replace('.jpg', '_bw.jpg').replace('.png', '_bw.png'))
        gray_img = Image.open(gray_img_name).convert('RGB')

        if self.transform:
            color_img = self.transform(color_img)
            gray_img = self.transform(gray_img)

        #imgs=torch.cat((color_img, gray_img), dim=0)
        return color_img, gray_img

if __name__ =='__main__':
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = CustomDataset(color_dataset_folder='Dataset/ColorImgs', gray_dataset_folder='Dataset/GrayImgs', transform=transform)