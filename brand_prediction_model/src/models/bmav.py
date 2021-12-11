import torch
from torch import nn
from torchvision import models
from efficientnet_pytorch import EfficientNet

class BMAV(nn.Module):
    '''
    Pretrained model backbone, two linear layers with dropout in between.
    View feature appended after FC of pretrained network
    '''
    def __init__(self, pretrained_backbone, in_features, num_classes, dropout_1, dropout_2): 
        '''
        Initializing 2 linear layers, dropout and Leaky RelU layers
        '''
        assert pretrained_backbone in ["resnet34", "resnet50", "densenet121", "efficientnet"], "This model only supports resnet34, resnet50, densenet121 and efficientnet"
        super(BMAV, self).__init__()
        
        if pretrained_backbone == "resnet34":
            self.backbone = models.resnet34(pretrained=True)
        elif pretrained_backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=True)
        elif pretrained_backbone == "densenet121":
            self.backbone = models.densenet121(pretrained=True)
        elif pretrained_backbone == "efficientnet":
            self.backbone = EfficientNet.from_pretrained("efficientnet-b0")

        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout_1 = nn.Dropout(dropout_1)
        self.dropout_2 = nn.Dropout(dropout_2)
        self.activation = nn.LeakyReLU()

    def forward(self, x): 
        '''
        Forward pass through network
        '''
        out = self.dropout_1(self.activation(self.backbone(x[0])))
        out = torch.cat([out, x[1]], dim=-1)
        out = self.activation(self.fc1(out))
        out = self.fc2(self.dropout_2(out))

        return out