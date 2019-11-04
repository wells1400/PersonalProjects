import os
import cv2
import random
#import torch
#import torchvision
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#import torch.utils.data as data
from PIL import Image
from tensorflow import keras
#from torch.utils.data import DataLoader


# 输入healthy 和bad图片主目录，训练测试划分比例
# 将healthy 和bad图片按比例进行分割，分别置于TrainData和TestData目录
class DataInitialize:
    def __init__(self, root_path=r'E:\OfficialWorkRemote\OfficialWork\SIOA\WheatClassification\SourceData',
                 train_test_rate=0.6, reshape_size=(512, 512), mode='train'):
        self.root_path = root_path
        self.image_path = self.root_path + r'/Images'
        self.train_test_rate = train_test_rate
        self.reshape_size = reshape_size

        self.train_data = []
        self.test_data = []
        self.mode = mode
        self.process_engine()

    # 获取bad和healthy图片路径list
    def image_path_getter(self):
        bad_images = os.listdir(self.image_path + r'/bad')
        bad_images = [os.path.join(self.image_path + r'/bad' + '/', images) for images in bad_images]
        healthy_images = os.listdir(self.image_path + r'/healthy')
        healthy_images = [os.path.join(self.image_path + '/healthy' + '/', images) for images in healthy_images]
        return bad_images, healthy_images

    # shffle训练和测试image
    def image_split_shuffle(self, bad_images, healthy_images):
        # 随机选择训练集的数据,rand_select_1是bad里面train抽取，rand_select_2是healthy里面train抽取，
        rand_select_1 = random.sample([i for i in range(len(bad_images))],
                                      round(self.train_test_rate * len(bad_images)))
        rand_select_2 = random.sample([i for i in range(len(bad_images))],
                                      round(self.train_test_rate * len(bad_images)))
        for index in range(len(bad_images)):
            if index in rand_select_1:
                self.train_data.append((bad_images[index], [0]))
            else:
                self.test_data.append((bad_images[index], [0]))
            if index in rand_select_2:
                self.train_data.append((healthy_images[index], [1]))
            else:
                self.test_data.append((healthy_images[index], [1]))

    # 图片reshape
    def image_reshape(self, image_path):
        image = cv2.imread(image_path)
        # 单通道图像
        image_b = cv2.resize(image[:, :, 0], self.reshape_size, interpolation=cv2.INTER_CUBIC)
        image_g = cv2.resize(image[:, :, 1], self.reshape_size, interpolation=cv2.INTER_CUBIC)
        image_r = cv2.resize(image[:, :, 2], self.reshape_size, interpolation=cv2.INTER_CUBIC)
        # image转成tensor
        #image_b_tensor = tf.convert_to_tensor(image_b)
        #image_g_tensor = tf.convert_to_tensor(image_g)
        #image_r_tensor = tf.convert_to_tensor(image_r)
        return [image_b, image_g, image_r]

    # 数据处理主程
    def process_engine(self):
        bad_images, healthy_images = self.image_path_getter()
        self.image_split_shuffle(bad_images, healthy_images)

    def data_getter(self, item):
        image_path = item[0]
        image_label = item[1]
        res_image = self.image_reshape(image_path)
        return res_image, image_label

    # 迭代器,输入batch_size 输出样本（[image_b_tensor,image_g_tensor,image_r_tensor],image_label_tensor）
    def data_generator(self, batch_size):
        res_iamges = []
        res_labels = []
        pos_limit = len(self.train_data) if self.mode == 'train' else len(self.test_data)
        rand_select_index = random.sample([i for i in range(pos_limit)], batch_size)
        for item_index in rand_select_index:
            if self.mode == 'train':
                res_image, image_label = self.data_getter(self.train_data[item_index])
                res_iamges.append(res_image)
                res_labels.append(image_label)
                continue
            res_image, image_label = self.data_getter(self.test_data[item_index])
            res_iamges.append(res_image)
            res_labels.append(image_label)
        #coded_tensor = tf.one_hot(res_labels, depth=2)
        #array_b = coded_tensor.eval()
        return np.array(res_iamges), np.array(res_labels)


if __name__ == '__main__':
    data_test = DataInitialize(root_path=r'E:\OfficialWorkRemote\OfficialWork\SIOA\WheatClassification\SourceData',
                               train_test_rate=0.6)
    print(data_test.data_generator(2))
    tf.ten