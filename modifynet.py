import torch
import torch.nn as nn
from encoder import *
import torch.nn.functional as F

# modify source code of SFNet
class AlignedModule(nn.Module):
    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModule, self).__init__()
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 2, kernel_size=kernel_size, padding=1, bias=False)

    def forward(self, x):
        low_feature, h_feature = x
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)
        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        output = F.grid_sample(input, grid, align_corners=False)
        return output
    
class ModifyNet(nn.Module):
    def __init__(self):
        super(ModifyNet, self).__init__()
        self.e1 = ResBlock(3, 64)
        self.e2 = ResBlock(64, 128)
        self.e3 = ResBlock(128, 256)
        self.e4 = ResBlock(256, 512)
        self.ppm = PPM(512, 512)
        
        self.o4_1 = AlignedModule(256, 512)
        self.o4_2 = AlignedModule(128,512)
        self.o4_3 = AlignedModule(64, 512)
        
        self.o3_1 = AlignedModule(128, 256)
        self.o3_2 = AlignedModule(64, 256)
        
        self.o2_1 = AlignedModule(64, 128)
        
        self.conv0 = nn.Conv2d(960, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x1, p1 = self.e1(x)
        x2, p2 = self.e2(p1)
        x3, p3 = self.e3(p2)
        x4, p4 = self.e4(p3)
        
        pp = self.ppm(x4)
        
        f4_1 = self.o4_1((x3, pp))
        f4_2 = self.o4_2((x2, f4_1))
        s4_3 = self.o4_3((x1, f4_2))
        
        f3_1 = self.o3_1((x2, x3))
        s3_2 = self.o3_2((x1, f3_1))
        
        s2_1 = self.o2_1((x1, x2))
        
        s1 = x1
        s2 = s2_1
        s3 = s3_2
        s4 = s4_3
        
        # n c h w
        d0 = torch.cat([s1, s2, s3, s4], 1)
        d1 = self.conv0(d0)
        d2 = self.sigmoid(d1)
        return d2