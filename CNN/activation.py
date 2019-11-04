# -*- coding:utf-8 -*-
"""
激活函数的前向和反向过程
包括sigmoid、ReLU
"""
import numpy as np


def sigmoid_forward(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_backward(z):
    return np.multiply(z, 1 - z)


def relu_forward(z):
    """
    大于0则输出原值，小于或等于0则输出0
    """
    return np.maximum(0, z)


def relu_backward(next_dz, z):
    """
    若激活函数之前的神经元的值（z）大于0，则偏导（next_z）传播时在此层乘1，否则乘0
    where(判断条件，值1，值2)：若判断条件为真，则返回值1，否则返回值2
    greater(x,y)：比较x和y的大小，x>y则返回True
    """
    return np.where(np.greater(z, 0), next_dz, 0)
