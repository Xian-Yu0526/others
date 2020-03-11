import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,32,3,2,1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )#24+0.5
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,3,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )#12+0.5
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,128,3,2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )#5+0.5
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )  # 3
        self.conv5 = nn.Sequential(
            nn.Conv2d(256,2,3,1,0)
        )#1


    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        y4 = self.conv4(y3)
        y5 = self.conv5(y4)
        miu = y5[:,:1,:,:]
        sigma = y5[:,1:,:,:]
        return miu,sigma

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )  # 3
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )#5

        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2,0,1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )  # 12
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1,1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )#24
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 3, 2, 1, 1),
            nn.Tanh()
        )  # 48

    def forward(self, miu,log_sigma,z):
        x = z*torch.exp(log_sigma)+miu
        x = x.permute([0,3,1,2])

        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        y4 = self.conv4(y3)
        y5 = self.conv5(y4)

        return y5

class Net_total(nn.Module):
    def __init__(self):
        super(Net_total, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x,z):
        miu,log_sigma = self.encoder.forward(x)
        output = self.decoder.forward(miu,log_sigma,z)
        return miu,log_sigma,output