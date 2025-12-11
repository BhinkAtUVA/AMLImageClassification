import torch
from torch import nn
import torch.nn.functional as F

class GeM(nn.Module):
    """
    Generalized Mean Pooling.
    """
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = x.mean(dim=(-1, -2))  
        x = x.pow(1.0 / self.p)
        return x



class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.mean(dim=(-1, -2))      
        y = F.relu(self.fc1(y), inplace=True)
        y = torch.sigmoid(self.fc2(y)) 
        y = y.view(b, c, 1, 1)
        return x * y

class BasicModel(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, se_reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.down = None

        self.se = SEBlock(out_ch, reduction=se_reduction)

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)

        if self.down is not None:
            identity = self.down(x)

        out += identity
        out = F.relu(out, inplace=True)
        return out


class BetterBirdCNN(nn.Module):
    """
    model with SE blocks and GeM pooling.
    """
    def __init__(self, num_classes=200):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),                 
        )

        self.layer1 = nn.Sequential(
            BasicModel(64, 64, 1),
            BasicModel(64, 64, 1),
        )

        self.layer2 = nn.Sequential(
            BasicModel(64, 128, 2),
            BasicModel(128, 128, 1),
        )

        self.layer3 = nn.Sequential(
            BasicModel(128, 256, 2),
            BasicModel(256, 256, 1),
        )

        self.layer4 = nn.Sequential(
            BasicModel(256, 512, 2),
            BasicModel(512, 512, 1),
        )

        self.pool = GeM()
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)     
        x = self.layer1(x)    
        x = self.layer2(x)    
        x = self.layer3(x)    
        x = self.layer4(x)    
        x = self.pool(x)      
        x = self.drop(x)
        x = self.fc(x)        
        return x