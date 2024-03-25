from skimage.feature import local_binary_pattern
from scipy.spatial.distance import euclidean
import numpy as np
import cv2
import os
import torch
from utils import jload
from PIL import Image
from skimage import color
import instance_model
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import model
from data import norm_lab_to_rgb

offsets = (45.60540338353164, 2.561262893902535, 6.502379186955269)
scales  = (27.067869578097213, 12.064396872062012, 16.395105370918557)

def extract_features(segment):
    if isinstance(segment,torch.Tensor):
        segment=np.array(segment)
    average_brightness = np.mean(segment)
    
    lbp = local_binary_pattern(segment, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(257), density=True)
    
    contours, _ = cv2.findContours(segment.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    moments = cv2.moments(contours[0])
    hu_moments = cv2.HuMoments(moments).flatten()
    
    return np.concatenate([[average_brightness], lbp_hist, hu_moments])

def calculate_similarity(features1, features2):
    return euclidean(features1, features2)

def find_most_similar(features, reference_features, threshold):
    similarities = [calculate_similarity(features, ref) for ref in reference_features]
    most_similar_index = np.argmin(similarities)
    if similarities[most_similar_index] < threshold:
        return most_similar_index, similarities[most_similar_index]
    else:
        return None, None

segment_test_json=jload("autodl-tmp/segment_json/test.json")

colorized_image_path="autodl-tmp/dataset/test/43405.jpg"
ref_image_path="autodl-tmp/dataset/test/45664.jpg"

colorized_image_r=segment_test_json.get(colorized_image_path,None)
ref_image_r=segment_test_json.get(ref_image_path,None)

instance_cinn=instance_model.ColorizationCINN()
instance_model.eval_model_load(instance_cinn,"autodl-tmp/instance/resnet_transformer/instance_cinn.pt")
instance_cinn.cuda()
instance_cinn.eval()

cinn=model.ColorizationCINN()
model.eval_model_load(cinn,"autodl-tmp/instance_based/resnet_transformer/check_point/resnet_transformer_instance_based_20.pt")
cinn.cuda()
cinn.eval()

if colorized_image_r is not None and ref_image_r is not None:
    colorized_image_bboxes=[]
    ref_image_bboxes=[]

    for score,bbox in zip(colorized_image_r["scores"],colorized_image_r["bboxes"]):
        left,upper,right,lower = [int(x) for x in bbox]
        if (right-left)>=64 and (lower-upper)>=64 and score>0.5:
            colorized_image_bboxes.append(bbox)
    for score,bbox in zip(ref_image_r["scores"],ref_image_r["bboxes"]):
        left,upper,right,lower = [int(x) for x in bbox]
        if (right-left)>=64 and (lower-upper)>=64 and score>0.5:
            ref_image_bboxes.append(bbox)

    #生成各自的实例组
    all_colorized_image_instance = torch.zeros((len(colorized_image_bboxes),3,256,256)) 
    for idx,mask in enumerate(colorized_image_bboxes):
        img=Image.open(colorized_image_path)
        img=img.crop(mask) 
        img=img.resize((256,256))
        img=np.array(img)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=0)
        img=color.rgb2lab(img).transpose((2,0,1))
        img = torch.tensor(img).float()    
        all_colorized_image_instance[idx]=img
    for j in range(len(all_colorized_image_instance)):
        for i in range(3):
            all_colorized_image_instance[j][i] = (all_colorized_image_instance[j][i] - offsets[i]) / scales[i]
    
    all_ref_image_instance = torch.zeros((len(ref_image_bboxes),3,256,256)) 
    for idx,mask in enumerate(ref_image_bboxes):
        img=Image.open(colorized_image_path)
        img=img.crop(mask) 
        img=img.resize((256,256))
        img=np.array(img)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=0)
        img=color.rgb2lab(img).transpose((2,0,1))
        img = torch.tensor(img).float()    
        all_ref_image_instance[idx]=img
    for j in range(len(all_ref_image_instance)):
        for i in range(3):
            all_ref_image_instance[j][i] = (all_ref_image_instance[j][i] - offsets[i]) / scales[i]

    #按照L通道进行相似度计算
    z_instance=[]
    for idx,colorized_image_instance in enumerate(all_colorized_image_instance):
        colorized_image_feature=extract_features(colorized_image_instance[0]) 
        reference_features_list=[]
        for ref_image_instance in all_ref_image_instance:
            reference_features_list.append(extract_features(ref_image_instance[0]))
        threshold = 0.5
        most_similar_index, similarity = find_most_similar(colorized_image_feature, reference_features_list, threshold)
        if most_similar_index is not None:
            Lab=torch.tensor(all_ref_image_instance[most_similar_index])
            Lab=torch.unsqueeze(Lab,dim=0).cuda()
            z,_=instance_cinn(TF.resize(Lab,(256,256)))
            z_instance.append(z.cuda())
        else:
            z=torch.randn(1, instance_model.ndim_total).cuda()
            z_instance.append(z)
    z_instance=torch.stack(z_instance,dim=0).cuda()
    z_instance=torch.squeeze(z_instance,0)
    ab_gen,_ = instance_cinn.reverse_sample(z_instance, TF.resize(all_colorized_image_instance[:,:1],(256,256)).cuda())

    im=Image.open(colorized_image_path)
    w,h=im.size
    full_ab = np.zeros((2,h,w), dtype=np.float32)
    count_ab = np.zeros((2, h, w), dtype=np.float32)
    ab_gen=ab_gen.detach().cpu().numpy()
    for idx, (bbox, generated_ab) in enumerate(zip(colorized_image_bboxes,ab_gen)):
        left, upper, right, lower = [int(x) for x in bbox]
        ab_unsample=np.zeros((2,lower - upper,right - left))
        for i in range(2):
            ab_unsample[i]=cv2.resize(generated_ab[i], (right - left,lower - upper), interpolation=cv2.INTER_CUBIC)
            ab_unsample[i] = ab_unsample[i] * offsets[i+1] + scales[i+1]
        full_ab[:,upper:lower, left:right] +=ab_unsample
        count_ab[:,upper:lower, left:right]+=1
    epsilon = 1e-10
    full_ab /= (count_ab + epsilon)
    img=color.rgb2lab(np.array(im)).transpose(2,0,1)[:1]
    lab_image = np.concatenate([img, full_ab], axis=0)
    lab_image_rgb = color.lab2rgb(lab_image.transpose((1,2,0))) 
    plt.imsave('mid.jpg', lab_image_rgb) 
    L=img
    #colorized_image_path=""
    #ref_image_path=""
    colorized_image=Image.open('mid.jpg')
    colorized_image=colorized_image.resize((256,256))
    colorized_image=np.array(colorized_image).transpose((2,0,1))
    colorized_image=torch.tensor(colorized_image).float().cuda()
    colorized_image=torch.unsqueeze(colorized_image,dim=0)

    ref_image=Image.open(ref_image_path)

    ref_image=ref_image.resize((256,256))
    ref_image=np.array(ref_image)
    ref_image=color.rgb2lab(ref_image).transpose((2,0,1))
    ref_image = torch.tensor(ref_image).float()   
    for i in range(3):
        ref_image[i]=(ref_image[i]-offsets[i])/scales[i]
    ref_image=torch.tensor(ref_image).cuda()
    ref_image=torch.unsqueeze(ref_image,dim=0)
    z,_=cinn(ref_image,ref_image)
    ab_gen,_=cinn.reverse_sample(z,colorized_image)
    L=np.expand_dims(L,axis=0)
    rgb=norm_lab_to_rgb(L,ab_gen.cpu())
    for im in rgb:
        im=np.transpose(im, (1,2,0))
        plt.imsave("colorful_result.jpg", im)



