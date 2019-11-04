# -*- coding:utf-8 -*-
"""
CNN手写数字识别的训练过程
训练结束后将参数保存在序列化文件中
"""
import mnist
import pickle
from network import LeNet

l = LeNet()
x_data, y_data = mnist.train_load()
x_data = x_data[:1000,:]
y_data = y_data[:1000]
learning_rate = 0.01
batch_size = 320  # 每批次的训练样本数
num_x_data = x_data.shape[0]  # 总训练样本数
num_batch = num_x_data // batch_size  # 每轮训练批数：1875

for t in range(1):
    loss = 0
    print("第%d轮训练" % (t+1))
    for i in range(num_batch):
        x = x_data[i * batch_size:(i + 1) * batch_size, :]
        x = x.reshape(x.shape[0], 1, 28, 28)
        y = y_data[i * batch_size:(i + 1) * batch_size, :]
        l.forward(x)
        loss += l.backward(x, y)
        for weight in ["W1", "b1", "W2", "b2", "W3", "b3", "K1", "kb1", "K2", "kb2"]:
            l.weights[weight] -= learning_rate * l.gradients[weight]
        print("已训练%4d批" % (i + 1))
    print("经过%d轮的训练之后，loss值为: %f " % (t + 1, loss / num_x_data))

with open("model\\model_weights.pickle", "wb") as w:
    pickle.dump(l.weights, w)
