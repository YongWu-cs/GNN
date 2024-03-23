import torch.nn as nn
import torch
import config
import torchvision.models as models

class CondNet(nn.Module):
    '''conditioning network'''
    def __init__(self):
        super().__init__()

        class Flatten(nn.Module):
            def __init__(self, *args):
                super().__init__()
            def forward(self, x):
                return x.view(x.shape[0], -1)
        #特征提取网络引入transofomer作为全局特征
        #cnn进行局部特征
        self.resolution_levels = nn.ModuleList([
                           nn.Sequential(nn.Conv2d(3,  64, 3, padding=1,stride=2),
                                         nn.LeakyReLU(),
                                         nn.BatchNorm2d(64),
                                         nn.Conv2d(64, 64, 3, padding=1,stride=2)),

                           nn.Sequential(nn.LeakyReLU(),
                                         nn.BatchNorm2d(64),
                                         nn.Conv2d(64,  128, 3, padding=1),
                                         nn.LeakyReLU(),
                                         nn.BatchNorm2d(128),
                                         nn.Conv2d(128, 128, 3, padding=1, stride=2)),

                           nn.Sequential(nn.LeakyReLU(),
                                         nn.BatchNorm2d(128),
                                         nn.Conv2d(128, 128, 3, padding=1, stride=2)),

                           nn.Sequential(nn.LeakyReLU(),
                                         nn.AvgPool2d(4),
                                         Flatten(),
                                         )])
        in_feature=self.get_infeature()
        self.resolution_levels.pop(-1)
        self.resolution_levels.append(nn.Sequential(nn.LeakyReLU(),
                        nn.BatchNorm2d(128),
                        nn.AvgPool2d(4),
                        Flatten(),
                        nn.Linear(in_features=in_feature, out_features=512)
                        ))

    def get_trainable_parameters(self):
        return self.parameters()

    def get_condition_nodes(self):
        x = torch.randn(1, 3, config.image_h_w,config.image_h_w)
        outputs = [x]
        conditions = []
        for m in self.resolution_levels:
            res=m(outputs[-1])
            outputs.append(res)
            if len(res.size()) == 4:
                conditions.append((int(res.size(1)), int(res.size(2)), int(res.size(3))))
        conditions.append(512)
        print(conditions)
        return conditions

    def get_infeature(self):
        x = torch.randn(1, 3, config.image_h_w, config.image_h_w) 
        for layer in self.resolution_levels:
            x = layer(x)
        in_features = x.numel() / x.size(0)
        return int(in_features) 

    def forward(self, c):
        outputs = [c]
        for m in self.resolution_levels:
            outputs.append(m(outputs[-1]))
        return outputs[1:]

class resnet_backbone(nn.Module):
    def __init__(self):  
        super(resnet_backbone, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.avgpool = resnet.avgpool

        self.fc = nn.Linear(512, 512)
    
    #后面增加一个fc，可训练的，其余部分保持为不可训练
    def get_condition_nodes(self):
        features = []
        x=torch.randn(1,3, config.image_h_w, config.image_h_w)
        x = self.conv1(x)
        features.append((int(x.size(1)), int(x.size(2)), int(x.size(3))))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        features.append((int(x.size(1)), int(x.size(2)), int(x.size(3))))
        x = self.layer2(x)
        features.append((int(x.size(1)), int(x.size(2)), int(x.size(3))))
        x = self.layer3(x)
        #features.append((int(x.size(1)), int(x.size(2)), int(x.size(3))))
        x = self.layer4(x)
        #features.append((int(x.size(1)), int(x.size(2)), int(x.size(3))))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x=self.fc(x)
        features.append((int(x.size(1))))
        
        return features

    def get_trainable_parameters(self):
        return self.parameters()

    def forward(self, x):
        features = []
        x = self.conv1(x)
        features.append(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        features.append(x)
        
        return features
    
if __name__=="__main__":
    cond=resnet_backbone()
    features=cond.get_condition_nodes()
    x=torch.randn(1,3,256,256)
    y=cond(x)
    for i in y:
        print(y.shape)
    print(y)