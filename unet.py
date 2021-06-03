import torch
import torch.nn as nn
import torch.nn.functional as F
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1) 
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) 
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.conv9 = nn.Conv2d(512, 1024, 3, padding=1)
        self.conv10 = nn.Conv2d(1024, 1024, 3, padding=1)
        
        self.conv11 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv12 = nn.Conv2d(1024, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 3, padding=1)
        
        self.conv14 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv15 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv16 = nn.Conv2d(256, 256, 3, padding=1)
        
        self.conv17 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv18 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv19 = nn.Conv2d(128, 128, 3, padding=1)      

        self.conv20 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv21 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv22 = nn.Conv2d(64, 64, 3, padding=1) 
        
        self.conv23 = nn.Conv2d(64, 1, 1)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):      
        x = F.relu(self.conv2(F.relu(self.conv1(x))))
        x1_dup = x
        x = self.pool1(x)
        x = F.relu(self.conv4(F.relu(self.conv3(x))))
        x2_dup = x
        x = self.pool2(x) 
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x3_dup = x
        x = self.pool3(x)  
        x = F.relu(self.conv8(F.relu(self.conv7(x))))
        x4_dup = x
        x = self.pool4(x)          
        x = F.relu(self.conv10(F.relu(self.conv9(x))))
        
        x = self.conv11(x)
        x = torch.cat((x4_dup,x),1)

        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.conv14(x)
        x = torch.cat((x3_dup,x),1)
      
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))
        x = self.conv17(x)
        x = torch.cat((x2_dup,x),1)       
      
        x = F.relu(self.conv18(x))
        x = F.relu(self.conv19(x))
        x = self.conv20(x)
        x = torch.cat((x1_dup,x),1)    
       
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = self.conv23(x)
        
        x = self.sigmoid(x)
        return x