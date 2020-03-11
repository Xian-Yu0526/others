import torch.nn as nn
import torch

class D_net(nn.Module):
    def __init__(self):
        super(D_net,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.LeakyReLU(0.2)
        )  # 24+0.5
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,3,2,1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )#24+0.5
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,128,3,2,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )#12+0.5
        self.conv4 = nn.Sequential(
            nn.Conv2d(128,256,3,2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )#5+0.5
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )  # 3
        self.conv6 = nn.Sequential(
            nn.Conv2d(512,1,3,1,0),
            nn.Sigmoid()
        )#1

    def forward(self, x):

        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        y4 = self.conv4(y3)
        y5 = self.conv5(y4)
        y6=self.conv6(y5)
        y6=y6.reshape(y6.size(0),1)
        return y6

class G_net(nn.Module):
    def __init__(self):
        super(G_net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 512, 3, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )  # 3
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )#5

        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2,0,1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )  # 12
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1,1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )#24
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )  # 48
        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 3, 2, 1, 1),
            nn.Tanh()
        )  # 96

    def forward(self,x):

        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        y4 = self.conv4(y3)
        y5 = self.conv5(y4)
        y6=self.conv6(y5)

        return y6

