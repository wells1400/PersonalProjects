# -*- coding:utf-8 -*-
"""
损失函数
暂时只考虑均方误差
"""
import numpy as np


def mean_squared_loss(y_predict, y_true):
    loss = np.mean(0.5 * (np.square(y_predict - y_true)))  # 损失函数值
    dy = - (y_true - y_predict)  # 损失函数关于网络输出的梯度
    return loss, dy
