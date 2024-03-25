from instance_data import offsets as instance_offsets
from instance_data import scales as instance_scales
from instance_model import ColorizationCINN,eval_model_load,ndim_total
import config
from utils import jload
from PIL import Image
import numpy as np
from copy import deepcopy
import torch
import torchvision.transforms.functional as TF
from skimage import color
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import create_folder

def process_image(image_path, bboxes):
    # 打开图像并转换到LAB色彩空间
    img = Image.open(image_path).convert('LAB')
    
    img_np = np.array(img)
    
    # 分离L和ab通道
    L_channel, a_channel, b_channel = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
    
    a=np.zeros_like(L_channel)
    b=np.zeros_like(L_channel)
    if len(bboxes) !=0:
        for bbox in bboxes:
            a[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]=a_channel[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            b[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]=b_channel[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    LAB_cropped_padded = np.stack((L_channel, a, b), axis=-1)
    img_LAB_padded = Image.fromarray(LAB_cropped_padded, 'LAB')
    img_RGB_padded = img_LAB_padded.convert('RGB')
    
    return img_RGB_padded

# instance_cinn = ColorizationCINN()
# eval_model_load(instance_cinn,config.instance_model_load_path)
# instance_cinn.cuda()
# instance_cinn.eval()

segment_train_json=jload("autodl-tmp/segment_json/train.json")
segment_val_json=jload("autodl-tmp/segment_json/val.json")
segment_test_json=jload("autodl-tmp/segment_json/test.json")

data_path="autodl-tmp/dataset"
mode=["train","val","test"]
all_images={"train":[],"val":[],"test":[]}
for m in mode:
    path=data_path+"/"+m
    if path=="images":
        continue
    for subdir, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                all_images[m].append(os.path.join(subdir, file))
for key in all_images.keys():
    image_list=all_images[key]
    create_folder('autodl-tmp/ab_part_colorization/{}'.format(key))
    for img in tqdm(image_list):
        im=Image.open(img)

        image_name=os.path.basename(img)
        if key=="train":
            r=segment_train_json.get(img,None)
        elif key=="val":
            r=segment_val_json.get(img,None)
        else:
            r=segment_test_json.get(img,None)
        if r is not None:
            useful_mask = []
            for score, bbox in zip(r["scores"], r["bboxes"]):
                left,upper,right,lower = [int(x) for x in bbox]
                if (right-left)>=64 and (lower-upper)>=64 and score>0.5:
                    useful_mask.append(bbox)
            img=process_image(img,useful_mask)
        else:
            img=process_image(img,[])
        img.save('autodl-tmp/ab_part_colorization/{}/{}'.format(key,image_name))
        # if r is not None:
        #     useful_mask = []
        #     for score, bbox in zip(r["scores"], r["bboxes"]):
        #         left,upper,right,lower = [int(x) for x in bbox]
        #         if (right-left)>=64 and (lower-upper)>=64 and score>0.5:
        #             useful_mask.append(bbox)
        #     if len(useful_mask)==0:
        #         continue
        #     all_instance = torch.zeros((len(useful_mask),3,256,256)) 
        #     for idx,mask in enumerate(useful_mask):
        #         img=deepcopy(im)
        #         img=img.crop(mask) 
        #         img=img.resize((256,256))
        #         img=np.array(img)
        #         if img.ndim == 2:
        #             img = np.stack([img] * 3, axis=0)
        #         img=color.rgb2lab(img).transpose((2,0,1))
        #         img = torch.tensor(img).float()    
        #         all_instance[idx]=img
            
        #     for j in range(len(all_instance)):
        #         for i in range(3):
        #             all_instance[j][i] = (all_instance[j][i] - instance_offsets[i]) / instance_scales[i]
        #     try:
        #         L= all_instance[:, :1].cuda()
        #         z=torch.randn(all_instance.shape[0], ndim_total).cuda()
        #         ab_gen,_ = instance_cinn.reverse_sample(z, TF.resize(L,(config.image_h_w,config.image_h_w)))

        #         w,h=im.size
        #         full_ab = np.zeros((2,h,w), dtype=np.float32)
        #         ab_gen=ab_gen.detach().cpu().numpy()
        #         for idx, (bbox, generated_ab) in enumerate(zip(useful_mask,ab_gen)):
        #             left, upper, right, lower = [int(x) for x in bbox]
        #             ab_unsample=np.zeros((2,lower - upper,right - left))
        #             for i in range(2):
        #                 ab_unsample[i]=cv2.resize(generated_ab[i], (right - left,lower - upper), interpolation=cv2.INTER_CUBIC)
        #                 ab_unsample[i] = ab_unsample[i] * instance_scales[i+1] + instance_offsets[i+1]
        #             full_ab[:,upper:lower, left:right] = ab_unsample
        #         img=color.rgb2lab(np.array(im)).transpose(2,0,1)[:1]
        #         lab_image = np.concatenate([img, full_ab], axis=0)
        #         lab_image_rgb = color.lab2rgb(lab_image.transpose((1,2,0))) 
        #         plt.imsave('autodl-tmp/ab_part_colorization/{}/{}'.format(key,image_name), lab_image_rgb) 
        #     except:
        #         continue