import torch
import torch.nn as nn
from DropBlock import DropBlock2D

# Use two 3x3 conv as a 5x5 conv
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True)
    )   


class UNet(nn.Module):
    # out_channel: number of output channels
    # in_channel : number of input channels
    # n_layer : number of unet layers
    # s: number of output channel for the 1st conv layer
    def __init__(self, out_channel, in_channel, n_layer, s):
        super().__init__()
        
        #conv down path list
        self.convdown  =  nn.ModuleList()   
        self.convdown.append(double_conv(in_channel, s))
        for i in range(0,n_layer-1):
            self.convdown.append(double_conv(s, 2*s))
            s = s*2   
        
        self.conv_drop = DropBlock2D()
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
       
        #conv up path list
        self.convdup   = nn.ModuleList() 
        for i in range(0,n_layer-1):
            self.convdup.append(double_conv(int(1.5*s), int(s/2)))
            s = int(s/2) 
       
        self.conv_last = nn.Conv2d(s, out_channel, 1)
        
        
    def forward(self, x):
        # create a conv list for later concat two layers in the same level
        conv = []
        for i, down in enumerate(self.convdown):
            x = down(x)
            if i != len(self.convdown) - 1:
                conv.append(x)
                x = self.maxpool(self.conv_drop(x))
            else:
                x = self.upsample(self.conv_drop(x))        
                x = torch.cat([x, conv[len(conv)-1]], dim=1)
                
        
        for i, up in enumerate(self.convdup):
            x = up(x)
            if i != len(self.convdup) - 1:
                x = self.upsample(self.conv_drop(x))
                # concat two layers in the same level
                x = torch.cat([x, conv[len(conv)-i-2]], dim=1)
            
        out = self.conv_last(x)
        
        return out