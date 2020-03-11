import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from data_load import Data_load
import torch.utils.data as Data
import net
import matplotlib.pyplot as plt

if __name__ == '__main__':
    if not os.path.exists("./result"):
        os.makedirs("./result")


    dataset = Data_load(r"./xy_data")
    data1 = Data.DataLoader(dataset=dataset, batch_size=100, shuffle=True)


    net = net.Net_total().cuda()

    if os.path.exists(r'./moudle\p.pt'):
        try:
            net.load_state_dict(torch.load(r'./moudle\p.pt'))
            print("加载成功")
        except:
            print("加载不成功！")

    opt = torch.optim.Adam(net.parameters())
    loss_fn = nn.MSELoss(reduction="sum")

    count=0
    while True:
        count+=1
        for i, (imgs) in enumerate(data1):
            img=imgs.cuda()
            z = torch.randn(512).cuda()
            miu,log_sigma,out_img = net(img,z)
            en_loss = torch.mean((-torch.log(log_sigma**2)+miu**2+log_sigma**2-1)*0.5)
            de_loss = loss_fn(out_img,img)
            loss = en_loss+de_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            # print(loss.item())
            if i%20== 0:
                fake_img = out_img.data
                # fake_img = out_img.data.permute([0,2,3,1])
                # plt.imshow(fake_img[0].reshape(28,28))
                                # plt.pause(1)
                img = imgs.data
                save_image(fake_img,"./result/{}-fake_img.png".format(count+i),nrow=10)
                save_image(img,"./result/{}-real_img.png".format(count+i),nrow=10)
        torch.save(net.state_dict(),"./moudle/p.pt")
        print("第{}批损失为{}".format(count,loss.item()))
