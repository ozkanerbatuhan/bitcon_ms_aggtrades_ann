import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_size=40, num_classes=3):
        super(SimpleMLP, self).__init__()
        
        # Architecture: 40 -> 64 -> 32 -> 16 -> 3
        # ReLU activations
        
        self.layer1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        
        self.layer2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        
        self.layer3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.relu3 = nn.ReLU()
        
        self.layer4 = nn.Linear(16, num_classes)
        # No Softmax here! CrossEntropyLoss includes LogSoftmax.
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.layer2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        out = self.layer3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        
        out = self.layer4(out)
        return out
