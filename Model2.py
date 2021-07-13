from torch import nn as nn

class SimpleRegressionModel(nn.Module):
    def __init__(self):
        super(SimpleRegressionModel,self).__init__()
        self.fc1 = nn.Linear(1,1)
        self.fc2 = nn.Linear(1,1)

    def forward(self,mc):
        y = nn.functional.sigmoid(self.fc1(mc))
        y = self.fc2(y)
        return y