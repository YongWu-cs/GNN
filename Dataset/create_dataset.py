from mmdet.apis import DetInferencer
from tqdm import tqdm
import torch
import os

# folder_path = 'raw_dataset'
# all_images = []
# for file in os.listdir(folder_path):
#     if file.endswith(".jpg") or file.endswith(".png"):
#         all_images.append(os.path.join(folder_path, file))

# print(len(all_images))


paths=["celebeA/ColorImgs"]
all_images=[]
for path in paths:
    # 遍历每个子文件夹
    for subdir, _, files in os.walk(path):
        # 遍历每个文件
        for file in files:
            # 检查文件扩展名是否为图片
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                # 将图片路径加入相应的列表
                all_images.append(os.path.join(subdir, file))
print(len(all_images))

from PIL import Image
for img_path in all_images:
    img = Image.open(img_path)
    a=img.getbands()
    if len(a)<3:
        img.close()
        os.remove(img_path)
        print(img_path)

# folder_path = 'raw_dataset'
# all_images = []
# for file in os.listdir(folder_path):
#     if file.endswith(".jpg") or file.endswith(".png"):
#         all_images.append(os.path.join(folder_path, file))

# print(len(all_images))

paths=["celebeA/ColorImgs"]
all_images=[]
for path in paths:
    # 遍历每个子文件夹
    if path=="images":
        continue
    for subdir, _, files in os.walk(path):
        # 遍历每个文件
        for file in files:
            # 检查文件扩展名是否为图片
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                # 将图片路径加入相应的列表
                all_images.append(os.path.join(subdir, file))
print(len(all_images))
# inferencer = DetInferencer(model='solov2_r101-dcn_fpn_ms-3x_coco',device='cuda')

import shutil
# batch_size=24

# person_image=[]
# for idx in tqdm(range(0,len(all_images),batch_size)):
#     start=idx
#     end=min(idx+batch_size,len(all_images))
#     batch_file=all_images[start:end]
#     res=inferencer(batch_file, 
#                out_dir="autodl-tmp/segmentation_dataset",batch_size=batch_size)
#     for i,r in enumerate(res['predictions']):
#         if 0 in r['labels']:
#             img=Image.open(batch_file[i])
#             img_width, img_height = img.size
#             human_boxes = [] 
#             for box, label in zip(r['bboxes'], r['labels']):
#                 if label == 0: 
#                     human_boxes.append(box)
#             tag=False
#             for box in human_boxes:
#                 box_width = box[2] - box[0]
#                 box_height = box[3] - box[1]
#                 box_area = box_width * box_height
#                 img_area = img_width * img_height
#                 if box_area >= img_area / 6:
#                     person_image.append(all_images[start+i])           
#                     tag=True
#                     break  
#             img.close()
#             if tag==False:
#                 os.remove(batch_file[i])
#                 print("remove {}".format(all_images[start+i]))
#     torch.cuda.empty_cache()

all_images=[]
for path in paths:
    # 遍历每个子文件夹
    for subdir, _, files in os.walk(path):
        # 遍历每个文件
        for file in files:
            # 检查文件扩展名是否为图片
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                # 将图片路径加入相应的列表
                all_images.append(os.path.join(subdir, file))
print(len(all_images))

train_dir="autodl-tmp/celebeA/train"
val_dir="autodl-tmp/celebeA/val"
test_dir="autodl-tmp/celebeA/test"
for dir in [train_dir,val_dir,test_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)

total_samples = len(all_images)
train_samples = int(0.9 * total_samples)
val_samples = int(0.05* total_samples)

train_images = all_images[:train_samples]
val_images = all_images[train_samples:train_samples + val_samples]
test_images = all_images[train_samples + val_samples:]

num=1
for image in train_images:
    new_filename = f"{str(num).zfill(6)}.jpg"
    destination_path = os.path.join(train_dir, new_filename)
    shutil.copy(image, destination_path)
    num+=1

print(num)

for image in val_images:
    new_filename = f"{str(num).zfill(6)}.jpg"
    destination_path = os.path.join(val_dir, new_filename)
    shutil.copy(image, destination_path)
    num+=1
print(num)

for image in test_images:
    new_filename = f"{str(num).zfill(6)}.jpg"
    destination_path = os.path.join(test_dir, new_filename)
    shutil.copy(image, destination_path)
    num+=1

print(num)