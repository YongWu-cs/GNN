import os
import shutil
import random

# 定义源文件夹和目标文件夹路径
source_folder = "Dataset/ColorImgs"
train_folder = "Dataset/Train/ColorImgs"
test_folder = "Dataset/Test/ColorImgs"

# 创建目标文件夹
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# 获取源文件夹下的所有文件
file_names = os.listdir(source_folder)

# 随机打乱文件顺序
random.shuffle(file_names)

# 计算分割点
split_point = int(0.7 * len(file_names))

# 划分训练集和测试集
train_files = file_names[:split_point]
test_files = file_names[split_point:]

# 移动文件到训练集文件夹
for file in train_files:
    source_file = os.path.join(source_folder, file)
    destination_file = os.path.join(train_folder, file)
    shutil.move(source_file, destination_file)

# 移动文件到测试集文件夹
for file in test_files:
    source_file = os.path.join(source_folder, file)
    destination_file = os.path.join(test_folder, file)
    shutil.move(source_file, destination_file)

print("文件已成功移动到训练集和测试集文件夹中。")
