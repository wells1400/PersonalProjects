# -*- coding:utf-8 -*-
"""
CNN识别手写数字 网络结构
"""
import numpy as np
from loss import mean_squared_loss
from structure import fc_forward, fc_backward, conv_forward, conv_backward, max_pooling_forward, max_pooling_backward
from activation import relu_forward, relu_backward, sigmoid_forward, sigmoid_backward
from tools import toOneRow, backToNRow

np.random.seed(2019)


class LeNet(object):
    def __init__(self):
        # 存放参数值
        self.weights = {}

        # 注意范围
        self.weights["K1"] = 0.1 * (2 * np.random.random((1, 3, 5, 5)) - 1)
        self.weights["kb1"] = np.zeros(3) + 1

        self.weights["K2"] = 0.1 * (2 * np.random.random((3, 6, 5, 5)) - 1)
        self.weights["kb2"] = np.zeros(6) + 1

        self.weights["W1"] = 0.1 * (2 * np.random.random((4 * 4 * 6, 24)) - 1)
        self.weights["b1"] = np.zeros(24) + 1

        self.weights["W2"] = 0.1 * (2 * np.random.random((24, 24)) - 1)
        self.weights["b2"] = np.zeros(24) + 1

        self.weights["W3"] = 0.1 * (2 * np.random.random((24, 10)) - 1)
        self.weights["b3"] = np.zeros(10) + 1

        # 存放神经元的值
        self.nurons = {}

        # 存放梯度
        self.gradients = {}

    def forward(self, train_data):
        # 前向传播，卷积层→展开成一行→全连接
        self.nurons["c1"] = conv_forward(train_data, self.weights["K1"], self.weights["kb1"])
        self.nurons["c1_relu"] = relu_forward(self.nurons["c1"])
        self.nurons["p1"] = max_pooling_forward(self.nurons["c1_relu"], (2, 2))
        self.nurons["c2"] = conv_forward(self.nurons["p1"], self.weights["K2"], self.weights["kb2"])
        self.nurons["c2_relu"] = relu_forward(self.nurons["c2"])
        self.nurons["p2"] = max_pooling_forward(self.nurons["c2_relu"], (2, 2))

        self.nurons["p2_"] = toOneRow(self.nurons["p2"])  # 展开成一行

        self.nurons["h1"] = fc_forward(self.nurons["p2_"], self.weights["W1"], self.weights["b1"])
        self.nurons["h1_relu"] = relu_forward(self.nurons["h1"])
        self.nurons["h2"] = fc_forward(self.nurons["h1_relu"], self.weights["W2"], self.weights["b2"])
        self.nurons["h2_relu"] = relu_forward(self.nurons["h2"])
        self.nurons["y"] = fc_forward(self.nurons["h2_relu"], self.weights["W3"], self.weights["b3"])
        self.nurons["y_sigmoid"] = sigmoid_forward(self.nurons["y"])

        return self.nurons["y_sigmoid"]

    def backward(self, train_data, y_true):
        # 反向传播，全连接→展开还原→卷积层
        loss, self.gradients["y_sigmoid"] = mean_squared_loss(self.nurons["y_sigmoid"], y_true)
        self.gradients["y"] = sigmoid_backward(self.gradients["y_sigmoid"])
        self.gradients["W3"], self.gradients["b3"], self.gradients["h2_relu"] = fc_backward(self.gradients["y"],
                                                                                            self.weights["W3"],
                                                                                            self.nurons["h2_relu"])
        self.gradients["h2"] = relu_backward(self.gradients["h2_relu"], self.nurons["h2"])
        self.gradients["W2"], self.gradients["b2"], self.gradients["h1_relu"] = fc_backward(self.gradients["h2"],
                                                                                            self.weights["W2"],
                                                                                            self.nurons["h1_relu"])
        self.gradients["h1"] = relu_backward(self.gradients["h1_relu"], self.nurons["h1"])
        self.gradients["W1"], self.gradients["b1"], self.gradients["p2_"] = fc_backward(self.gradients["h1"],
                                                                                        self.weights["W1"],
                                                                                        self.nurons["p2_"])

        self.gradients["p2"] = backToNRow(self.gradients["p2_"], self.nurons["p2"])

        self.gradients["c2_relu"] = max_pooling_backward(self.gradients["p2"], self.nurons["c2_relu"], (2, 2))
        self.gradients["c2"] = relu_backward(self.gradients["c2_relu"], self.nurons["c2"])
        self.gradients["K2"], self.gradients["kb2"], self.gradients["p1"] = conv_backward(self.gradients["c2"],
                                                                                          self.weights["K2"],
                                                                                          self.nurons["p1"])
        self.gradients["c1_relu"] = max_pooling_backward(self.gradients["p1"], self.nurons["c1_relu"], (2, 2))
        self.gradients["c1"] = relu_backward(self.gradients["c1_relu"], self.nurons["c1"])
        self.gradients["K1"], self.gradients["kb1"], _ = conv_backward(self.gradients["c1"],
                                                                       self.weights["K1"],
                                                                       train_data)

        return loss
