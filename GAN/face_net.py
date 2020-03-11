import torch.nn as nn
import torch

class D_net(nn.Module):
    def __init__(self):
        super(D_net,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2,bias=False),
            nn.LeakyReLU(0.2)
        )  # 47
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,4,2,1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )#23
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,128,4,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )#11
        self.conv4 = nn.Sequential(
            nn.Conv2d(128,256,4,2,1,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )#5
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1,bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )  # 2
        self.conv6 = nn.Sequential(
            nn.Conv2d(512,1,2,2,0,bias=False),
            nn.Sigmoid()
        )#1

    def forward(self, x):
        y1 = self.conv1(x)
        # print(y1.shape)
        y2 = self.conv2(y1)
        # print(y2.shape)
        y3 = self.conv3(y2)
        # print(y3.shape)
        y4 = self.conv4(y3)
        # print(y4.shape)
        y5 = self.conv5(y4)
        # print(y5.shape)
        y6 = self.conv6(y5)
        # print(y6.shape)
        y6=y6.reshape(y6.size(0),1)
        return y6

class G_net(nn.Module):
    def __init__(self):
        super(G_net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 512, 2, 2, 0,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )  # 2
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2,1,1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )#5

        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2,1,1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )  # 11
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1,1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )#23
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1, 1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )  # 47
        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 4, 2, 0, 0,bias=False),
            nn.Tanh()
        )  # 96

    def forward(self,x):

        y1 = self.conv1(x)
        # print(y1.shape)
        y2 = self.conv2(y1)
        # print(y2.shape)
        y3 = self.conv3(y2)
        # print(y3.shape)
        y4 = self.conv4(y3)
        # print(y4.shape)
        y5 = self.conv5(y4)
        # print(y5.shape)
        y6=self.conv6(y5)
        # print(y6.shape)

        return y6

