image_h_w=256
ab_h_w=image_h_w//4
lr=3e-4
batch_size=36
val_step=400

train_drop_last=True
cond_net_trainable=True
cond_net="resnet_transformer"#["cond_net","resnet_18","resnet_50","resnet_transformer","resnet50_transformer","resnet34_transformer"]
#sub_conv=[64,128,256,256,512] #resnet_18
sub_conv=[64,128,256,256,512] 

N_epochs=120
run_name="resnet_transformer_instance"
loss={"NLL":1}
init_scale=0.030 
clip_norm=60

save_check_point_epoch=10
save_check_point_num=5
model_save_path="autodl-tmp/instance/resnet34_transformer"

model_load_path=None
dataset_path="autodl-tmp/segment_dataset"
use_log=True

mode="train"