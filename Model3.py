from torch import nn as nn
import torch
import torchvision

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel,self).__init__()
        resnet18 = torchvision.models.resnet18(pretrained = True)
        self.model = nn.Sequential(*(list(resnet18.children())[:-1]),nn.Flatten())
        # MC path
        self.fc1 = nn.Linear(1,1)
        self.fc2 = nn.Linear(1,1)
        # Image path
        self.fc3 = nn.Linear(512,1)
        self.fc4 = nn.Linear(2,1)
        
    def forward(self,image,mitotic_count):
        # Image path
        x1 = self.model(image)
        x1 = nn.functional.sigmoid(self.fc3(x1))
        
        # MC path
        x2 = nn.functional.sigmoid(self.fc1(mitotic_count))
        x2 = self.fc2(x2)
        # Concatenate the image and the MC path
        x = torch.cat((x1,x2), dim = 1)
        y = self.fc4(x)
        
        return y