import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import config
import torch.nn.init as init
from utils import create_folder
from cond_net import CondNet,resnet_backbone

ndim_total=2*config.ab_h_w*config.ab_h_w
#分为实例网络和全局网络，全局网络使用transformer提取到的特征，实例网络使用cnn提取到的特征
class ColorizationCINN(nn.Module):
    '''cINN, including the ocnditioning network'''
    def __init__(self,optimizer=None):
        super().__init__()

        if config.cond_net=="cond_net":
            self.cond_net = CondNet()
            self.cond_feautre=self.cond_net.get_condition_nodes()
        
        if config.cond_net=="resnet_18":
            self.cond_net=resnet_backbone()
            self.cond_feautre=self.cond_net.get_condition_nodes()

        self.cinn = self.build_inn(self.cond_feautre)

        self.trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        for p in self.trainable_parameters:
            p.data = 0.02 * torch.randn_like(p)

        if config.cond_net_trainable:
            self.trainable_parameters += list(self.cond_net.get_trainable_parameters())
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.trainable_parameters, config.lr)
        else:
            self.set_optimizer(optimizer)
    
    def set_optimizer(self,optimizer):
        self.optimizer=optimizer

    def build_inn(self,cond_features):

        def sub_conv(ch_hidden, kernel):
            pad = kernel // 2
            return lambda ch_in, ch_out: nn.Sequential(
                                            nn.Conv2d(ch_in, ch_hidden, kernel, padding=pad),
                                            nn.ReLU(),
                                            nn.Conv2d(ch_hidden, ch_out, kernel, padding=pad))

        def sub_fc(ch_hidden):
            return lambda ch_in, ch_out: nn.Sequential(
                                            nn.Linear(ch_in, ch_hidden),
                                            nn.ReLU(),
                                            nn.Linear(ch_hidden, ch_out))

        nodes = [Ff.InputNode(2, config.ab_h_w, config.ab_h_w)]
        # outputs of the cond. net at different resolution levels
        conditions = [Ff.ConditionNode(cond_features[0][0],cond_features[0][1],cond_features[0][2]),
                      Ff.ConditionNode(cond_features[1][0],cond_features[1][1],cond_features[1][2]),
                      Ff.ConditionNode(cond_features[2][0],cond_features[2][1],cond_features[2][2]),
                      #Ff.ConditionNode(cond_features[3][0],cond_features[3][1],cond_features[3[2]),
                      #Ff.ConditionNode(cond_features[4])
                      Ff.ConditionNode(cond_features[3])]

        split_nodes = []

        # 1
        subnet = sub_conv(32, 3)
        for k in range(4):
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':1.0},
                                 conditions=conditions[0]))

        nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'rebalance':0.5}))

        # 2
        for k in range(6):
            subnet = sub_conv(64, 3 if k%2 else 1)

            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':1.0},
                                 conditions=conditions[1]))
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed':k}))

        nodes.append(Ff.Node(nodes[-1], Fm.Split,
                             {'section_sizes':[2,6], 'dim':0}))
        split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))
        nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'rebalance':0.5}))

        # 3
        for k in range(6):
            subnet = sub_conv(128, 3 if k%2 else 1)
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':0.6},
                                 conditions=conditions[2]))
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed':k}))
        nodes.append(Ff.Node(nodes[-1], Fm.Split,
                             {'section_sizes':[4,4], 'dim':0}))
        split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))
        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))
        
        # for k in range(6):
        #     subnet = sub_conv(128, 3 if k%2 else 1)
        #     nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
        #                          {'subnet_constructor':subnet, 'clamp':0.6},
        #                          conditions=conditions[2]))
        #     nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed':k}))
        # nodes.append(Ff.Node(nodes[-1], Fm.Split,
        #                      {'section_sizes':[4,4], 'dim':0}))
        # split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))
        # nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))

        # 4
        subnet = sub_fc(512)
        for k in range(8):
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':0.6},
                                 conditions=conditions[3]))
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed':k}))

        # concat everything
        nodes.append(Ff.Node([s.out0 for s in split_nodes] + [nodes[-1].out0],
                             Fm.Concat1d, {'dim':0}))
        nodes.append(Ff.OutputNode(nodes[-1]))

        return Ff.ReversibleGraphNet(nodes + split_nodes + conditions, verbose=False)


    def forward(self, Lab):
        #lab取ab通道
        ab=Lab[:,1:]
        ab = F.interpolate(ab, size=(config.ab_h_w,config.ab_h_w))
        l=Lab[:,:1]
        l=l.repeat(1, 3, 1, 1)
        z,jac = self.cinn(ab, c=self.cond_net(l),jac=True)
        return z, jac

    def reverse_sample(self, z, L):
        L=L.repeat(1,3,1,1)
        return self.cinn(z, c=self.cond_net(L), rev=True)

def eval_model_load(model,model_path):
    try:
        checkpoint = torch.load(model_path)
        state_dict = {k:v for k,v in checkpoint['model_state_dict'].items() if 'tmp_var' not in k}
    except:
        state_dict = {k:v for k,v in torch.load(model_path).items() if 'tmp_var' not in k}
    state_dict_true={}
    for k1,k2 in zip(model.state_dict().keys(),state_dict.keys()):
        if k1!=k2:
            print(k1,k2)
            state_dict_true[k1]=state_dict[k2]
        else:
            state_dict_true[k2]=state_dict[k2]
    model.load_state_dict(state_dict_true)

def model_load(model,optimizer,scheduler,model_path):
    checkpoint = torch.load(model_path)
    state_dict = {k:v for k,v in checkpoint['model_state_dict'].items() if 'tmp_var' not in k}
    state_dict_true={}

    for k1,k2 in zip(model.state_dict().keys(),state_dict.keys()):
        if k1!=k2:
            print(k1,k2)
            state_dict_true[k1]=state_dict[k2]
        else:
            state_dict_true[k2]=state_dict[k2]
    model.load_state_dict(state_dict_true)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    loss= checkpoint['loss']
    return epoch,step,loss

def model_save(model,epoch,step,optimizer,scheduler,loss):
    import os
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'step':step,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    create_folder(f'{config.model_save_path}/check_point')
    checkpoints = [file for file in os.listdir(f'{config.model_save_path}/check_point') if file.endswith('.pt')]
    if len(checkpoints) > config.save_check_point_num:
        # 根据文件名排序，假设文件名反映了保存时间
        checkpoints.sort()
        num_to_delete = len(checkpoints) - config.save_check_point_num
        for i in range(num_to_delete):
            os.remove(os.path.join(f'{config.model_save_path}/check_point', checkpoints[i]))
            print(f"Old checkpoint {checkpoints[i]} has been deleted.")
    torch.save(checkpoint, f'{config.model_save_path}/check_point/{config.run_name}_{epoch}.pt')
    print("Checkpoint saved")

if __name__=="__main__":
    iterations_per_epoch = 25000/ config.batch_size
    N_epochs = 100
    T_max = iterations_per_epoch * 10 / 2 
    cINN=ColorizationCINN()
    print(cINN.state_dict().keys())
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(cINN.optimizer, T_max=T_max)
    # optimizer=cINN.optimizer
    # # test=torch.randn([1,3,config.image_h_w,config.image_h_w])
    # # z=cINN(test)
    # model_save(cINN,10,20,optimizer,scheduler,-2.0)
    # load_path='output/lsun_cinn_checkpoint_epoch_10.pt'
    # epoch,step,loss=model_load(cINN,optimizer,scheduler,load_path)
    # print(epoch,step,loss)