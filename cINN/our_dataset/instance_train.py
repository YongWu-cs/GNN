import instance_config as config
config.mode="train"
from time import time
from tqdm import tqdm
import torch
import torch.optim
import numpy as np
import math
import torch.nn.functional as F
import instance_model as model
import instance_data as data
from loss import NLL_loss,MSE_loss
import wandb

if __name__=="__main__":
    torch.manual_seed(42)
    config_dict = config.__dict__
    config_dict = {k: v for k, v in config_dict.items() if not k.startswith('_')}
    if config.use_log:
        wandb.init(project='GNN',
            name=config.run_name,
            config=config_dict)
    torch.multiprocessing.freeze_support()
    cinn = model.ColorizationCINN()
    if config.use_log:
        wandb.watch(cinn,log="all")
    
    optimizer=cinn.optimizer
    N_epochs = config.N_epochs
    T_max = 16
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(cinn.optimizer, T_max=T_max)
    start_epoch=0
    start_step=0
    cinn.to("cuda")
    if config.model_load_path is not None:
        start_epoch,start_step,loss=model.model_load(cinn,optimizer,scheduler,config.model_load_path)
        print("load model success")
        cinn.set_optimizer(optimizer)
    
    t_start = time()
    nll_mean = []

    total_steps = N_epochs * (len(data.train_loader))
    
    progress_bar = tqdm(total=total_steps, desc='Training Progress')
    steps_per_epoch =  len(data.train_loader) 

    step=steps_per_epoch*start_epoch+start_step
    progress_bar.update(step)
    for epoch in range(start_epoch,N_epochs):
        for i, Lab in enumerate(data.train_loader):
            if start_step>0:
                start_step-=1
                continue

            L=Lab[:,:1].cuda()
            ab=Lab[:,1:]

            Lab = Lab.cuda()
            z, log_j = cinn(Lab)

            nll_loss=0
            mse_loss=0
            
            if "NLL" in config.loss.keys():
                nll_loss=config.loss["NLL"]*NLL_loss(z,log_j,model.ndim_total)
            if "MSE" in config.loss.keys():
                ab_gen,_= cinn.reverse_sample(z, L)
                ab = F.interpolate(ab, size=(config.ab_h_w,config.ab_h_w))
                mse_loss=config.loss["MSE"]*MSE_loss(ab_gen.cpu(),ab)
            total_loss = nll_loss+mse_loss

            total_loss.backward()
            nll_mean.append(total_loss.item())
            total_norm=torch.nn.utils.clip_grad_norm_(cinn.trainable_parameters, config.clip_norm)
            if config.use_log:
                wandb.log({"un_clip_grad_norm": total_norm.item()})
                wandb.log({"clip_grad_norm": min(total_norm.item(),config.clip_norm)})
            cinn.optimizer.step()
            cinn.optimizer.zero_grad()

            if i % config.val_step==0:
                with torch.no_grad():
                    val_loss=0
                    for _, Lab in enumerate(data.val_loader):
                        L=Lab[:,:1].cuda()
                        ab=Lab[:,1:]
                        Lab = Lab.cuda()
                        z, log_j = cinn(Lab)
                        nll_val_loss=0
                        mse_val_loss=0
                        if "NLL" in config.loss.keys():
                            nll_val_loss=config.loss["NLL"]*NLL_loss(z,log_j,model.ndim_total)
                        if "MSE" in config.loss.keys():
                            ab_gen,_= cinn.reverse_sample(z, L)
                            ab = F.interpolate(ab, size=(config.ab_h_w,config.ab_h_w))
                            mse_val_loss=config.loss["MSE"]*MSE_loss(ab_gen.cpu(),ab)
                        val_loss=val_loss+nll_val_loss+mse_val_loss
                    val_loss=val_loss/len(data.val_loader)
                step+=config.val_step
                if config.use_log:
                    wandb.log({"train_loss": np.mean(nll_mean), "val_loss": val_loss,"lr":cinn.optimizer.param_groups[0]['lr'], "step": step})    
                nll_mean = []
            progress_bar.update(1)
        if config.use_log:
            wandb.log({"epoch": epoch})
        scheduler.step()
        if epoch%config.save_check_point_epoch==0:
            optimizer=cinn.optimizer
            model.model_save(cinn,epoch,0,optimizer,scheduler,np.mean(nll_mean))
            print("auto save success")
    torch.save(cinn.state_dict(), f'{config.model_save_path}/instance_cinn.pt')
    if config.use_log:
        wandb.finish()