import os
import shutil
from PIL import Image
from tqdm import tqdm

color_dataset_folder = "Dataset/Train/ColorImgs"
gray_dataset_folder = "Dataset/Train/GrayImgs"

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(gray_dataset_folder):
    os.makedirs(gray_dataset_folder)
else:
    # 清空目标文件夹
    shutil.rmtree(gray_dataset_folder)
    os.makedirs(gray_dataset_folder)

# 循环处理输入文件夹中的每张彩色照片
for filename in tqdm(os.listdir(color_dataset_folder)):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # 打开彩色照片
        color_img_path = os.path.join(color_dataset_folder, filename)
        color_img = Image.open(color_img_path)
        
        # 转换为黑白照片
        gray_img = color_img.convert("L")
        
        # 获取新的文件名
        new_filename = filename.split(".")[0] + "_bw.jpg"
        new_img_path = os.path.join(gray_dataset_folder, new_filename)
        
        # 保存黑白照片
        gray_img.save(new_img_path)

print("所有照片转换完成.")