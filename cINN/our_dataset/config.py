image_h_w=256
ab_h_w=image_h_w//4
lr=1e-3
batch_size=36
train_drop_last=True
cond_net_trainable=True
cond_net="resnet_transformer"#["cond_net","resnet_18","resnet_transformer"]
sub_conv=[64,128,256,512,512]
val_step=500

N_epochs=120
run_name="resnet_transformer_full_veiw"
loss={"NLL":1}
init_scale=0.030 
clip_norm=10

save_check_point_epoch=10
save_check_point_num=2
model_save_path="autodl-tmp/full_veiw/resnet_transformer"

model_load_path=None
instance_mode=True
dataset_path="autodl-tmp/dataset"
use_log=False

mode="train"