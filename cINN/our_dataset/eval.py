import config
config.mode="eval"
from os.path import join
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import model
import data
import os
import wandb
from PIL import Image
import torchvision.transforms as T
from utils import create_folder
def colorize_test_set(cinn,temp=1., num=8, img_folder='base_model_result'):
    for i in range(num):
        torch.manual_seed(42*i+42)
        img_path=img_folder+f"_{i}"
        create_folder(img_path)
        with torch.no_grad():
            for (Lab,ref,img_p) in tqdm(data.test_loader):
                img_name=[]
                for p in img_p:
                    filename = os.path.basename(p)
                    img_name.append(f"{filename}")

                Lab = Lab.cuda()
                L = Lab[:, :1]
                ref=ref.cuda()
                if config.instance_mode==False:
                    ref=L.repeat(1,3,1,1)
                z = temp * torch.randn(Lab.shape[0], model.ndim_total).cuda()
                

                ab_gen,_ = cinn.reverse_sample(z, ref)
                
                rgb_gen = data.norm_lab_to_rgb(L.cpu(), ab_gen.cpu())

                for im,name in zip(rgb_gen,img_name):
                    im = np.transpose(im, (1,2,0))
                    plt.imsave(join(img_path, name), im)

def best_of_n(n,img_folder,test_folder="autodl-tmp/dataset/test"):
    '''computes the best-of-n MSE metric''' 
    all_image_name = []
    for file in os.listdir(test_folder):
        if file.endswith(".jpg") or file.endswith(".png"):
            filename = os.path.basename(file)
            all_image_name.append(filename)
    errors=np.zeros([len(all_image_name),n])
    progress_bar = tqdm(total=len(all_image_name)*n, desc='Training Progress')
    for i in range(num):
        img_path=img_folder+f"_{i}"
        for idx,name in enumerate(all_image_name):
            file_1=test_folder+"/"+name
            file_2=img_path+"/"+name

            img_1=np.array(Image.open(file_1)).transpose((2,0,1))
            img_1=torch.tensor(img_1)
            img_1=T.Resize(size=(config.image_h_w,config.image_h_w))(img_1)
            img_1=np.array(img_1).transpose((1,2,0))
            img_2=np.array(Image.open(file_2))
            errors[idx,i]=np.mean((img_1 - img_2) ** 2)
            progress_bar.update(1)
    min_errors = np.min(errors, axis=1)
    print(F'MSE best of {n}')
    print(np.sqrt(np.mean(min_errors)))
    return np.sqrt(np.mean(min_errors))

def rgb_var(n,img_folder,test_folder="autodl-tmp/dataset/test"):
    '''computes the pixel-wise variance of samples'''
    all_image_name = []
    for file in os.listdir(test_folder):
        if file.endswith(".jpg") or file.endswith(".png"):
            filename = os.path.basename(file)
            all_image_name.append(filename)

    all_var=[]
    for idx,name in enumerate(all_image_name):
        image=[]
        for i in range(8):
            img_path=img_folder+f"_{i}"
            img_name=img_path+"/"+name
            img=np.array(Image.open(img_name)).reshape(1,-1)
            image.append(img)
        all_var.append(np.mean(np.var(image, axis=0)))
    print(F'Var (of {n} samples)')
    print(np.mean(all_var))
    print(F'sqrt(Var) (of {n} samples)')
    print(np.sqrt(np.mean(all_var)))

    return np.mean(all_var),np.sqrt(np.mean(all_var))

from pytorch_fid.fid_score import calculate_fid_given_paths
def FID(n,img_folder,test_folder="autodl-tmp/dataset/test"):
    all_image_name = []
    base_name=[]
    for file in os.listdir(test_folder):
        if file.endswith(".jpg") or file.endswith(".png"):
            all_image_name.append(os.path.join(test_folder, file))
            filename = os.path.basename(file)
            base_name.append(filename)
    temp_path=test_folder+"_temp"
    create_folder(path=temp_path)
    for all_name,name in zip(all_image_name,base_name):
        img_1=np.array(Image.open(all_name)).transpose((2,0,1))
        img_1=torch.tensor(img_1)
        img_1=T.Resize(size=(config.image_h_w,config.image_h_w))(img_1)
        img_1=np.array(img_1).transpose((1,2,0))
        plt.imsave(join(temp_path, name), img_1)

    res=[]
    for i in range(n):
        fid_value = calculate_fid_given_paths([temp_path, img_folder+"_"+f"{i}"],
                                        batch_size=50,
                                        device='cuda',  # 或者 'cpu'，取决于你的系统配置
                                        dims=2048)
        print(fid_value)
        res.append(fid_value)
    return res

def record_compare_pic(wandb,n,img_folder,test_folder="autodl-tmp/dataset/test"):
    all_image_name = []
    base_name=[]
    for file in os.listdir(test_folder):
        if file.endswith(".jpg") or file.endswith(".png"):
            all_image_name.append(os.path.join(test_folder, file))
            filename = os.path.basename(file)
            base_name.append(filename)
    
    import random
    random_numbers = random.sample(range(len(all_image_name)), 10)
    random_all_image_name = [all_image_name[i] for i in random_numbers]
    random_base_name = [base_name[i] for i in random_numbers]

    colums=["ground truth"]+["generate {}".format(i) for i in range(n)]
    
    for idx in tqdm(range(len(random_base_name))):
        table = wandb.Table(columns=colums)
        record=[]
        base_pic=Image.open(random_all_image_name[idx])
        base_pic=base_pic.resize((128,128))
        record.append(wandb.Image(base_pic))
        for i in range(n):
            img_path=img_folder+"_"+f"{i}"+"/"+random_base_name[idx]
            img=Image.open(img_path)
            img=img.resize((128,128))
            record.append(wandb.Image(img))
        table.add_data(*record)
        wandb.log({"{}".format(random_base_name[idx]).format(idx): table})

if __name__=="__main__":
    model_path="autodl-tmp/continue_train_in_all_area/resnet_transformer/check_point/resnet_transformer_new_full_view_model_30.pt"
    img_folder='autodl-tmp/instance_aware/temp/result'
    test_folder=config.dataset_path+"/test"
    num=8
    
    use_log=False
    
    if use_log:
        wandb.init(project='GNN',
            name=config.run_name+"_eval")
    cinn = model.ColorizationCINN()
    model.eval_model_load(cinn,model_path)
    cinn.cuda()
    cinn.eval()
    colorize_test_set(cinn,temp=1,num=num,img_folder=img_folder)

    table = wandb.Table(columns=["type", "value"])
    mse_bset=best_of_n(num,img_folder,test_folder=test_folder)
    table.add_data("MSE","{:.2f}".format(mse_bset))
    var,sqrt_var=rgb_var(8,img_folder,test_folder=test_folder)
    table.add_data("var","{:.2f}".format(var))
    table.add_data("sqrt_var","{:.2f}".format(sqrt_var))
    fid=FID(num,img_folder,test_folder=test_folder)
    for idx,f in enumerate(fid):
        table.add_data(f"FID_{idx}","{:.2f}".format(f))
    mean_fid = np.mean(fid)
    variance_fid = np.var(fid)
    table.add_data(f"FID","{:.2f}±{:.2f}".format(mean_fid,variance_fid))
    if use_log:
        wandb.log({"eval_table": table})
        record_compare_pic(wandb,num,img_folder,test_folder)
        wandb.finish()
    print("eval sucessful")
