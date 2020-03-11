import torch
import torch.nn as nn
from data_load import Data_load
import torch.utils.data as Data
from torchvision.utils import save_image
import os
from  face_net import G_net,D_net

if __name__ == '__main__':

    if not os.path.exists("./test_face_img"):
        os.makedirs("./test_face_img")

    g_net = G_net().cuda()
    if os.path.exists(r'./face_params/p1.pt'):
        try:
            g_net.load_state_dict(torch.load(r'./face_params/p1.pt'))
            print("加载成功")
        except:
            print("加载不成功！")


            '''训练生成器'''
        z = torch.randn(1, 256, 1, 1).cuda()
        fake_img = g_net(z)


        fake_img = fake_img.data.reshape(-1,3,96,96)
        save_image(fake_img,"./test_face_img/{}-fake_img.jpg".format(1),normalize=True,scale_each=True)