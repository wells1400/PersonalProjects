# -*- coding:utf-8 -*-
"""
工具类
"""
import numpy as np


def oneHot(a):
    """
    将输入的（N,1）矩阵转换为（N,10）矩阵
    假设输入矩阵的第0行元素值为7，那么输出矩阵的第0行的第7个元素值为1，其他元素值为0
    """
    n = np.zeros((a.shape[0], 10))
    for i in range(a.shape[0]):
        n[i][int(a[i])] = 1.0
    return n


def toOneRow(z):
    """
    :param z: 形状为（N,D,H,W）
    :return: 形状为（N,D*H*W）
    """
    next_z = z.reshape((z.shape[0], z.shape[1] * z.shape[2] * z.shape[3]))
    return next_z


def backToNRow(next_z, z):
    """
    :param next_z: 形状为（N,D*H*W）
    :return: 形状为（N,D,H,W）
    """
    back_z = next_z.reshape((z.shape[0], z.shape[1], z.shape[2], z.shape[3]))
    return back_z
