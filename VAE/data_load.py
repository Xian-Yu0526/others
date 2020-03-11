import os
import PIL.Image as Img
import numpy as np
import torch.utils.data as Data



class Data_load(Data.Dataset):
    def __init__(self,path):
        self.img_path = []

        for name in os.listdir(path):
            data_path = os.path.join(path,name)

            self.img_path.append(data_path)
    def __len__(self):
        return len(self.img_path)
    def __getitem__(self, index):
        img_path1=self.img_path[index]
        data=np.array(Img.open(img_path1),dtype=np.float32).transpose([2,0,1])/255-0.5
        return data


