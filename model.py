from config import config
import torch.nn.functional as F
import torch 
import torchvision

class BasicConv2d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
def get_net(model_name):
    print("using default model : resnet101!")
    model = torchvision.models.resnet101(pretrained=True)
    model.conv1 = torch.nn.Conv2d(config.channels, 64, kernel_size=7, stride=2, padding=3,bias=False)
    model.fc = torch.nn.Linear(2048,config.num_classes)
    return model
