import torch
import torch.nn as nn
import torch.nn.functional as F

class AODNet(nn.Module):
    def __init__(self):
        super(AODNet, self).__init__()
        
      
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2, bias=True)
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3, bias=True)
        
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        
        cat1 = torch.cat((x1, x2), 1)
        x3 = self.relu(self.conv3(cat1))
        
        cat2 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.conv4(cat2))
        
        cat3 = torch.cat((x1, x2, x3, x4), 1)
        
        k = torch.sigmoid(self.conv5(cat3)) 
        

        clean_image = k * x - k + 1.0
        
        clean_image = F.relu(clean_image)
        clean_image = torch.min(clean_image, torch.tensor(1.0))
        
        return clean_image