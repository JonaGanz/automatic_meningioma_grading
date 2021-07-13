from torch import nn as nn
import torch
import torchvision

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel,self).__init__()
        resnet18 = torchvision.models.resnet18(pretrained = True)
        self.model = nn.Sequential(*(list(resnet18.children())[:-1]),nn.Flatten())
        # Image path
        self.fc1 = nn.Linear(512,1)
        self.fc2 = nn.Linear(1,1)
        
    def forward(self,image):
        # Image path
        x = self.model(image)
        y = self.fc1(x)
        
        return y