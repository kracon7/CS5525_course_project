import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MushroomNet(nn.Module):
    def __init__(self, num_classes):
        super(MushroomNet, self).__init__()

        self.num_classes = num_classes
        self.resnet = models.resnet18(pretrained=True)

        self.predictor = nn.Sequential(
                nn.Linear(1000, 256),
                nn.ReLU(),
                nn.Linear(256, self.num_classes),
                nn.Softmax(dim=1)
            )
        
    def forward(self, x):
        ft = self.resnet(x)
        out = self.predictor(ft)
        return out
