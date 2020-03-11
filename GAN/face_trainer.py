import torch
import torch.nn as nn
from data_load import Data_load
import torch.utils.data as Data
from torchvision.utils import save_image
import os
from  face_net import G_net,D_net

if __name__ == '__main__':
    batch_size = 100
    num_epoch = 100
    if not os.path.exists("./face_img"):
        os.makedirs("./face_img")
    if not os.path.exists("./face_params"):
        os.makedirs("./face_params")

    dataset = Data_load(r"F:\faces")
    data = Data.DataLoader(dataset=dataset, batch_size=100, shuffle=True,num_workers=4)

    d_net = D_net().cuda()
    g_net = G_net().cuda()
    if os.path.exists(r'./face_params/p0.pt'):
        try:
            d_net.load_state_dict(torch.load(r'./face_params/p0.pt'))
            g_net.load_state_dict(torch.load(r'./face_params/p1.pt'))
            print("加载成功")
        except:
            print("加载不成功！")

    loss_fn = nn.BCELoss().cuda()

    d_opt = torch.optim.Adam(d_net.parameters(),lr=0.0002,betas=(0.5, 0.999))
    g_opt = torch.optim.Adam(g_net.parameters(),lr=0.0002,betas=(0.5, 0.999))

    for epoch in range(num_epoch):
        for i,(img) in enumerate(data):
            # if epoch%5==0:
            real_img = img.cuda()
            real_label = torch.ones(img.size(0),1).cuda()
            fake_label = torch.zeros(img.size(0),1).cuda()

            real_out = d_net(real_img)
            d_loss_real = loss_fn(real_out,real_label)

            z = torch.randn(img.size(0),256,1,1).cuda()

            fake_img = g_net(z)
            fake_out = d_net(fake_img)
            d_loss_fake = loss_fn(fake_out,fake_label)

            d_loss = d_loss_real+d_loss_fake
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            '''训练生成器'''
            z = torch.randn(img.size(0), 256, 1, 1).cuda()
            fake_img = g_net(z)
            g_fake_out = d_net(fake_img)
            g_loss = loss_fn(g_fake_out,real_label)

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            if i%100== 0:
                print("Epoch:{}/{},d_loss:{},"
                      "g_loss:{},d_real:{},d_fake:{}"
                      .format(epoch,num_epoch,d_loss.item(),g_loss.item(),real_out.data.mean(),fake_out.data.mean()))
                torch.save(d_net.state_dict(),"./face_params/p0.pt")
                torch.save(g_net.state_dict(), "./face_params/p1.pt")
                print("保存成功")
                real_img = real_img.data.reshape(-1,3,96,96)
                fake_img = fake_img.data.reshape(-1,3,96,96)
                save_image(real_img,"./face_img/{}-real_img.jpg".format(epoch+i),nrow=10,normalize=True,scale_each=True)
                save_image(fake_img,"./face_img/{}-fake_img.jpg".format(epoch+i),nrow=10,normalize=True,scale_each=True)