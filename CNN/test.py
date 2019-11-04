# -*- coding:utf-8 -*-
"""
CNN手写数字识别的测试过程
模型参数保存在序列化文件中，在此读取
"""
import numpy as np
import mnist
import pickle
from network import LeNet


def get_accuracy(self, test_data, y_true):
    y_pre = self.forward(test_data)
    acc = np.mean(np.argmax(y_pre, axis=1) == np.argmax(y_true, axis=1))
    return acc

lenet = LeNet()

x_test, y_test = mnist.test_load()

with open("model\\model_weights.pickle", "rb") as r:
    lenet.weights = pickle.load(r)

acc = get_accuracy(lenet, x_test, y_test)

print("准确率为：", acc)
