image_h_w=512
ab_h_w=image_h_w//4
lr=3e-4
batch_size=36
val_step=500

train_drop_last=True
cond_net_trainable=True
cond_net="cond_net"#["cond_net","resnet_18"]

N_epochs=120
run_name="cond_net_celebA_model"
loss={"NLL":1}
init_scale=0.030 
clip_norm=60

save_check_point_epoch=10
save_check_point_num=2
model_save_path="autodl-tmp/celebeA_model"

model_load_path=None
dataset_path="autodl-tmp/celebeA"
use_log=True

mode="train"