from PIL import Image
import cv2
import numpy as np
import os
from tqdm import tqdm
import torch
from utils import create_folder,load_all_file_name,jdump,jload
from mmdet.apis import DetInferencer

def process_image(image_path, bbox, expand_ratio=2):
    # 打开图像并转换到LAB色彩空间
    img = Image.open(image_path).convert('LAB')
    
    # 计算扩大后的bbox区域，确保不超过图像边界
    bbox_expanded = [max(0, bbox[0] - (bbox[2] - bbox[0]) * (expand_ratio - 1) / 2),
                     max(0, bbox[1] - (bbox[3] - bbox[1]) * (expand_ratio - 1) / 2),
                     min(img.width, bbox[2] + (bbox[2] - bbox[0]) * (expand_ratio - 1) / 2),
                     min(img.height, bbox[3] + (bbox[3] - bbox[1]) * (expand_ratio - 1) / 2)]
    
    # 转换为numpy数组
    img_np = np.array(img)
    
    # 分离L和ab通道
    L_channel, a_channel, b_channel = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
    
    # 对L通道进行裁剪和padding
    L_cropped_padded = np.zeros_like(L_channel)
    L_cropped = L_channel[int(bbox_expanded[1]):int(bbox_expanded[3]), int(bbox_expanded[0]):int(bbox_expanded[2])]
    L_cropped_padded[int(bbox_expanded[1]):int(bbox_expanded[3]), int(bbox_expanded[0]):int(bbox_expanded[2])] = L_cropped
    
    # 对ab通道进行裁剪和padding
    ab_cropped_padded = np.zeros_like(img_np[:,:,1:])
    ab_cropped = img_np[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), 1:]
    ab_cropped_padded[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :] = ab_cropped
    
    # 重构LAB图像并转换回RGB
    LAB_cropped_padded = np.stack((L_cropped_padded, ab_cropped_padded[:,:,0], ab_cropped_padded[:,:,1]), axis=-1)
    img_LAB_padded = Image.fromarray(LAB_cropped_padded, 'LAB')
    img_RGB_padded = img_LAB_padded.convert('RGB')
    
    return img_RGB_padded

if __name__=="__main__":
    folder_path=['train','val','test']
    #batch_size = 24
    num=0
    #inferencer = DetInferencer(model='solov2_r101-dcn_fpn_ms-3x_coco', device='cuda')

    json_folder_dir="autodl-tmp/segment_json"

    for path in folder_path:
        folder_path = "autodl-tmp/dataset/"+path
        save_path="autodl-tmp/segment_dataset/"+path
        create_folder(save_path)
        json_dir=json_folder_dir+"/"+path+".json"
        all_images=load_all_file_name(folder_path)
        json_file=jload(json_dir)
        for image in tqdm(all_images):
            r=json_file.get(image,None)
            if r is None:
                continue

            for score,bbox in zip(r["scores"],r["bboxes"]):
                if score<0.75:
                    continue
                left,upper,right,lower = [int(x) for x in bbox]
                if (right-left)>=256 and (lower-upper)>=256:
                    img=Image.open(image)
                    img=img.crop(bbox)
                    img=img.resize((256,256),resample=Image.BICUBIC)
                    new_filename = f"{str(object=num).zfill(6)}.jpg"
                    img.save(save_path+"/"+new_filename)
                    num+=1

        # save_path="autodl-tmp/instance_dataset/"+path+".json"
        # create_folder(save_path)

        # start=0       
        # for idx in tqdm(range(start, len(all_images), batch_size)):
        #     start = idx
        #     end = min(idx + batch_size, len(all_images))
        #     batch_files = all_images[start:end]
        #     res = inferencer(batch_files,out_dir="autodl-tmp/segment", batch_size=batch_size)

        #     result_dict={}
        #     for i, r in enumerate(res['predictions']):
        #         result_dict[batch_files[i]]=r
        #         high_conf_instances=[]
        #         for j,score in enumerate(r["scores"]):
        #             if score>=0.6:
        #                 high_conf_instances.append(j)
        #             else:
        #                 break
        #         if len(high_conf_instances)==0:
        #             continue
        #         else:
        #             for sub_idx in high_conf_instances:
        #                 if sub_idx>=3:
        #                     break
        #                 # mask=r['masks'][sub_idx]
        #                 # binary_mask=mask_util.decode(mask)
                        
        #                 bbox = r['bboxes'][sub_idx]

        #                 img=process_image(batch_files[i],bbox)
        #                 # left,upper,right,lower = [int(x) for x in bbox]
        #                 # cropped_binary_mask = binary_mask[upper:lower , left:right]
        #                 # filled_image=Image.fromarray(mirror_fill_image(pic,cropped_binary_mask))
        #                 #has counts and size
        #                 # cropped_image=filled_image.resize((64,64))
        #                 new_filename = f"{str(object=num).zfill(6)}.jpg"
        #                 img.save(save_path+"/"+"{}".format(new_filename))
        #                 num+=1
        #     torch.cuda.empty_cache()
        #     jdump(result_dict,f"{json_save_dir}/{path}.json",mode="a")
        # jdump("]",f"{json_save_dir}/{path}.json",mode="a",is_end=True)