import torchvision
import torch.nn.functional as F 
from torch import nn
from config import config
class MyModel(nn.Module):
    def __init__(self,pretrained_model):
        super(MyModel,self).__init__()
        self.pretrained_model = pretrained_model
        self.layer1 = pretrained_model.layer1
        self.layer2 = pretrained_model.layer2
        self.layer3 = pretrained_model.layer3
        self.layer4 = pretrained_model.layer4

        pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(pretrained_model.fc.in_features,config.num_classes)
        )
        pretrained_model.fc = self.classifier

    def forward(self,x):
        return F.softmax(self.pretrained_model(x))
    

def get_net():
    #return MyModel(torchvision.models.resnet101(pretrained = True))
    model = torchvision.models.resnet50(pretrained = True)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(2048,config.num_classes)
    return model

