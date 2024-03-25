import torch.nn as nn
import torch
import instance_config as config
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
                                         nn.Conv2d(128,  256, 3, padding=1),
                                         nn.LeakyReLU(),
                                         nn.BatchNorm2d(num_features=256),
                                         nn.Conv2d(256, 128, 3, padding=1, stride=2)),

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

    def get_condition_nodes(self):
        features = []
        x=torch.randn(1,3, config.image_h_w, config.image_h_w)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        features.append((int(x.size(1)), int(x.size(2)), int(x.size(3))))
        x = self.layer2(x)
        features.append((int(x.size(1)), int(x.size(2)), int(x.size(3))))
        x = self.layer3(x)
        features.append((int(x.size(1)), int(x.size(2)), int(x.size(3))))
        x = self.layer4(x)
        features.append((int(x.size(1)), int(x.size(2)), int(x.size(3))))
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
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x=self.fc(x)
        features.append(x)
        
        return features

class resnet50_backbone(nn.Module):
    def __init__(self):  
        super(resnet50_backbone, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # Changed to resnet50

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.avgpool = resnet.avgpool

        self.fc = nn.Linear(2048, 1024)  # Changed input features to 2048

    def get_condition_nodes(self):
        features = []
        x = torch.randn(1, 3, config.image_h_w, config.image_h_w)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        features.append((int(x.size(1)), int(x.size(2)), int(x.size(3))))
        x = self.layer2(x)
        features.append((int(x.size(1)), int(x.size(2)), int(x.size(3))))
        x = self.layer3(x)
        features.append((int(x.size(1)), int(x.size(2)), int(x.size(3))))
        x = self.layer4(x)
        features.append((int(x.size(1)), int(x.size(2)), int(x.size(3))))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        features.append((int(x.size(1))))
        
        return features

    def get_trainable_parameters(self):
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def forward(self, x):
        features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        features.append(x)
        
        return features

class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x
class EncoderBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(EncoderBlock, self).__init__()
        self.ln_1 = nn.LayerNorm(input_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.dropout = nn.Dropout(p=0.0)
        self.ln_2 = nn.LayerNorm(input_dim)
        self.mlp = MLPBlock(input_dim, hidden_dim, input_dim)

    def forward(self, src):
        src = src + self.self_attention(self.ln_1(src), self.ln_1(src), self.ln_1(src))[0]
        src = self.dropout(src)
        src = src + self.mlp(self.ln_2(src))
        return src
    
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(TransformerModel, self).__init__()
        self.encoder_layers = nn.ModuleList([EncoderBlock(input_dim, hidden_dim, num_heads) for _ in range(num_layers)])

    def forward(self, src):
        for layer in self.encoder_layers:
            src = layer(src)
        return src

class ResNetTransformer(nn.Module):
    def __init__(self, resnet_out_dim=512, transformer_input_dim=768):
        super(ResNetTransformer, self).__init__()
        resnet=models.resnet18(pretrained=True)
        self.adaptation_layer = nn.Linear(resnet_out_dim, transformer_input_dim)
        self.transformer = TransformerModel(input_dim=transformer_input_dim, hidden_dim=3072, num_heads=8, num_layers=1)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool


    def get_condition_nodes(self):
        features = []
        x=torch.randn(1,3, 256, 256)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        features.append((int(x.size(1)), int(x.size(2)), int(x.size(3))))
        x = self.layer2(x)
        features.append((int(x.size(1)), int(x.size(2)), int(x.size(3))))
        x = self.layer3(x)
        features.append((int(x.size(1)), int(x.size(2)), int(x.size(3))))
        x = self.layer4(x)
        features.append((int(x.size(1)), int(x.size(2)), int(x.size(3))))
        x=self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.adaptation_layer(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x=torch.squeeze(x,dim=1)
        features.append(int(x.size(1)))
        
        return features

    def forward(self, x):
        features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.adaptation_layer(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = torch.squeeze(x,dim=1)
        features.append(x)

        return features

    def get_trainable_parameters(self):
        for param in self.parameters():
            if param.requires_grad:
                yield param

class ResNet50Transformer(nn.Module):
    def __init__(self, resnet_out_dim=2048, transformer_input_dim=1024):
        super(ResNet50Transformer, self).__init__()
        resnet=models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.adaptation_layer = nn.Linear(resnet_out_dim, transformer_input_dim)
        self.transformer = TransformerModel(input_dim=transformer_input_dim, hidden_dim=3072, num_heads=8, num_layers=1)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool


    def get_condition_nodes(self):
        features = []
        x=torch.randn(1,3, 256, 256)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        features.append((int(x.size(1)), int(x.size(2)), int(x.size(3))))
        x = self.layer2(x)
        features.append((int(x.size(1)), int(x.size(2)), int(x.size(3))))
        x = self.layer3(x)
        features.append((int(x.size(1)), int(x.size(2)), int(x.size(3))))
        x = self.layer4(x)
        features.append((int(x.size(1)), int(x.size(2)), int(x.size(3))))
        x=self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.adaptation_layer(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x=torch.squeeze(x,dim=1)
        features.append(int(x.size(1)))
        
        return features

    def forward(self, x):
        features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.adaptation_layer(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = torch.squeeze(x,dim=1)
        features.append(x)

        return features

    def get_trainable_parameters(self):
        for param in self.parameters():
            if param.requires_grad:
                yield param

class ResNet34Transformer(nn.Module):
    def __init__(self, resnet_out_dim=512, transformer_input_dim=512):
        super(ResNet50Transformer, self).__init__()
        resnet=models.resnet34(weights=models.ResNet50_Weights.DEFAULT)
        self.adaptation_layer = nn.Linear(resnet_out_dim, transformer_input_dim)
        self.transformer = TransformerModel(input_dim=transformer_input_dim, hidden_dim=3072, num_heads=8, num_layers=1)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool


    def get_condition_nodes(self):
        features = []
        x=torch.randn(1,3, 256, 256)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        features.append((int(x.size(1)), int(x.size(2)), int(x.size(3))))
        x = self.layer2(x)
        features.append((int(x.size(1)), int(x.size(2)), int(x.size(3))))
        x = self.layer3(x)
        features.append((int(x.size(1)), int(x.size(2)), int(x.size(3))))
        x = self.layer4(x)
        features.append((int(x.size(1)), int(x.size(2)), int(x.size(3))))
        x=self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.adaptation_layer(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x=torch.squeeze(x,dim=1)
        features.append(int(x.size(1)))
        
        return features

    def forward(self, x):
        features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.adaptation_layer(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = torch.squeeze(x,dim=1)
        features.append(x)

        return features

    def get_trainable_parameters(self):
        for param in self.parameters():
            if param.requires_grad:
                yield param


if __name__=="__main__":
    cond=resnet_backbone()
    features=cond.get_condition_nodes()
    x=torch.randn(1,3,256,256)
    y=cond(x)
    for i in y:
        print(i.shape)