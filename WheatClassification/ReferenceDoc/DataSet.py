#coding=utf-8

import torch.utils.data as data
from PIL import Image
import os
import torch

#将路径下的所有图片打上标签，并生成一个列表保存
def list_data_roots(path):
    image_roots = []
    label = 0
    for filename in os.listdir(path):   #读取路径下所有文件夹
        filenamepath = os.path.join(path, filename)   #读取所有文件夹的路径
        #print(filenamepath)
        temp_label = label
        #print(temp_label)
        for img_name in os.listdir(filenamepath):
            temp_root = os.path.join(filenamepath, img_name)
            #print(temp_root)
            """
            这样写简单点
            # item = (temp_root, temp_label)
            # print(item)
            # image_roots.append(item)   
            """
            if os.path.isfile(temp_root):
                item = (temp_root, temp_label)
                image_roots.append(item)
            else:
                continue
        label += 1
    return image_roots

#加载图片
def image_loader(img_root):
    image = Image.open(img_root)
      #读取图片
    image = image.resize((224, 224), resample=Image.LANCZOS)
    return image

#转换成张量输入
def num_to_tensor(number):
    num = []
    num_label = torch.tensor(number)
    num.append(num_label)
    return num_label

class My_dataset(data.Dataset):
    #data_path:训练/测试数据所在目录
    #loader:图片加载函数
    #transform:加载后的图片转换处理
    #target_transform:加载后的标签转换处理
    def __init__(self, data_path, loader=image_loader, transform=None, target_transform=None):
        data_roots = list_data_roots(data_path)
        if len(data_roots) == 0:
            print("Error")
        self.root = data_path
        self.loader = loader
        self.data_roots = data_roots
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        temp_root, temp_label = self.data_roots[index]
        img_temp = self.loader(temp_root)
        if self.transform is not None:
            img_temp = self.transform(img_temp)
        if self.target_transform is not None:
            temp_label = self.target_transform(temp_label)

        # print("img_temp:", img_temp, "temp_label:", temp_label)
        # image_temp = torch.Tensor(img_temp)
        # label_temp = torch.Tensor(temp_label)
        return img_temp, temp_label


    def __len__(self):
        return len(self.data_roots)

# 用来验证
# path = 'E:/data/data/train_data'
# print(list_data_roots(path))




