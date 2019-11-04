# -*- coding:utf-8 -*-
"""
全连接、卷积、池化
"""
import numpy as np


def fc_forward(z, W, b):
    """
    全连接层的前向传播
    """
    return np.matmul(z, W) + b


def fc_backward(next_dz, W, z):
    """
    全连接层的反向传播
    next_dz: 下一层的梯度
    """
    N = z.shape[0]  # 该批次的样本数
    dz = np.dot(next_dz, W.T)  # 当前层的梯度
    dw = np.dot(z.T, next_dz)  # 当前层权重的梯度
    db = np.sum(next_dz, axis=0)  # 当前层偏置的梯度
    return dw / N, db / N, dz


def conv_forward(z, K, b, padding=(0, 0), strides=(1, 1)):
    """
    多通道卷积前向过程
    shape=(2,3,4,4)，则[0,0]表示的是第0个样本的第0个矩阵
    z: 特征图矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    K: 卷积核,形状(C,D,k1,k2), C为输入通道数，D为输出通道数
    b: 偏置,形状(D,)
    """
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant',
                           constant_values=0)
    N, _, height, width = padding_z.shape
    C, D, k1, k2 = K.shape
    assert (height - k1) % strides[0] == 0, '步长不为1时，步长必须刚好能够被整除'
    assert (width - k2) % strides[1] == 0, '步长不为1时，步长必须刚好能够被整除'
    # 卷积后的矩阵，样本数不变，通道数改变，行列改变
    conv_z = np.zeros((N, D, 1 + (height - k1) // strides[0], 1 + (width - k2) // strides[1]))
    for n in np.arange(N):  # 控制单样本
        for d in np.arange(D):  # 控制单通道
            for h in range(0, height - k1 + 1, strides[0]):
                for w in range(0, width - k2 + 1, strides[1]):
                    conv_z[n, d, h // strides[0], w // strides[1]] = np.sum(
                        padding_z[n, :, h:h + k1, w:w + k2] * K[:, d]) + b[d]
    return conv_z


def _remove_padding(z, padding):
    """
    移除padding
    z: (N,C,H,W)
    paddings: (p1,p2)
    """
    if padding[0] > 0 and padding[1] > 0:
        return z[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]  # 仍旧是切片操作
    elif padding[0] > 0:
        return z[:, :, padding[0]:-padding[0], :]
    elif padding[1] > 0:
        return z[:, :, :, padding[1]:-padding[1]]
    else:
        return z


def _insert_zeros(dz, strides):
    """
    进行零填充
    dz: (N,D,H,W)
    """
    _, _, H, W = dz.shape  # H，W为卷积输出层的高度和宽度
    pz = dz
    if strides[0] > 1:
        for h in np.arange(H - 1, 0, -1):
            for o in np.arange(strides[0] - 1):
                pz = np.insert(pz, h, 0, axis=2)
    if strides[1] > 1:
        for w in np.arange(W - 1, 0, -1):
            for o in np.arange(strides[1] - 1):
                pz = np.insert(pz, w, 0, axis=3)
    return pz


def conv_backward(next_dz, K, z, padding=(0, 0), strides=(1, 1)):
    """
    多通道卷积层的反向过程
    next_dz: 卷积输出层的梯度,(N,D,H,W),H,W为卷积输出层的高度和宽度
    K: 当前层卷积核，(C,D,k1,k2)
    z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    """
    N, C, H, W = z.shape
    C, D, k1, k2 = K.shape

    # 步长不为（1，1）时补零
    padding_next_dz = _insert_zeros(next_dz, strides)

    # 卷积核高度和宽度翻转180度
    flip_K = np.flip(K, (2, 3))

    # 利用np.swapaxes()方法，交换C,D为D,C；D变为输入通道数，C变为输出通道数
    swap_flip_K = np.swapaxes(flip_K, 0, 1)

    # 增加高度和宽度0填充
    ppadding_next_dz = np.lib.pad(padding_next_dz, ((0, 0), (0, 0), (k1 - 1, k1 - 1), (k2 - 1, k2 - 1)), 'constant',
                                  constant_values=0)
    dz = conv_forward(ppadding_next_dz, swap_flip_K, np.zeros((C,)))

    # 求卷积和的梯度dK
    swap_z = np.swapaxes(z, 0, 1)  # 变为(C,N,H,W)与
    dK = conv_forward(swap_z, padding_next_dz, np.zeros((D,)))

    # 偏置的梯度
    db = np.sum(np.sum(np.sum(next_dz, axis=-1), axis=-1), axis=0)  # 求和，保证形状为（D,）

    # 如果前向传播中有padding，则把padding去掉
    dz = _remove_padding(dz, padding)

    return dK / N, db / N, dz


def max_pooling_forward(z, pooling=(2, 2), strides=(2, 2), padding=(0, 0)):
    """
    最大池化前向过程
    z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    pooling: 池化大小为(k1,k2)
    """
    N, C, H, W = z.shape
    # 零填充
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant',
                           constant_values=0)

    # 输出的高度和宽度
    out_h = (H + 2 * padding[0] - pooling[0]) // strides[0] + 1
    out_w = (W + 2 * padding[1] - pooling[1]) // strides[1] + 1

    pool_z = np.zeros((N, C, out_h, out_w))

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                    pool_z[n, c, i, j] = np.max(padding_z[n, c,
                                                strides[0] * i:strides[0] * i + pooling[0],
                                                strides[1] * j:strides[1] * j + pooling[1]])
    return pool_z


def max_pooling_backward(next_dz, z, pooling=(2, 2), strides=(2, 2), padding=(0, 0)):
    """
    最大池化反向过程
    next_dz：损失函数关于最大池化输出的损失
    z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    pooling: 池化大小(k1,k2)
    """
    N, C, H, W = z.shape
    _, _, out_h, out_w = next_dz.shape

    # 零填充
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant',
                           constant_values=0)
    # 零填充后的梯度
    padding_dz = np.zeros_like(padding_z)

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                    # 找到最大值的那个元素坐标，将梯度传给这个坐标
                    flat_idx = np.argmax(padding_z[n, c,
                                         strides[0] * i:strides[0] * i + pooling[0],
                                         strides[1] * j:strides[1] * j + pooling[1]])
                    h_idx = strides[0] * i + flat_idx // pooling[1]
                    w_idx = strides[1] * j + flat_idx % pooling[1]
                    padding_dz[n, c, h_idx, w_idx] += next_dz[n, c, i, j]

    # 如果前向传播中有padding，则把padding去掉
    return _remove_padding(padding_dz, padding)


def avg_pooling_forward(z, pooling=(2, 2), strides=(2, 2), padding=(0, 0)):
    """
    平均池化前向过程
    z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    pooling: 池化大小(k1,k2)
    """
    N, C, H, W = z.shape
    # 零填充
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant',
                           constant_values=0)

    # 输出的高度和宽度
    out_h = (H + 2 * padding[0] - pooling[0]) // strides[0] + 1
    out_w = (W + 2 * padding[1] - pooling[1]) // strides[1] + 1

    pool_z = np.zeros((N, C, out_h, out_w))

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                    pool_z[n, c, i, j] = np.mean(padding_z[n, c,
                                                 strides[0] * i:strides[0] * i + pooling[0],
                                                 strides[1] * j:strides[1] * j + pooling[1]])

    return pool_z


def avg_pooling_backward(next_dz, z, pooling=(2, 2), strides=(2, 2), padding=(0, 0)):
    """
    平均池化反向过程
    next_dz：损失函数关于最大池化输出的损失
    z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    pooling: 池化大小(k1,k2)
    """
    N, C, H, W = z.shape
    _, _, out_h, out_w = next_dz.shape
    # 零填充
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant',
                           constant_values=0)
    # 零填充后的梯度
    padding_dz = np.zeros_like(padding_z)

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                    # 每个神经元均分梯度
                    padding_dz[n, c,
                    strides[0] * i:strides[0] * i + pooling[0],
                    strides[1] * j:strides[1] * j + pooling[1]] += next_dz[n, c, i, j] / (pooling[0] * pooling[1])

    # 如果前向传播中有padding，则把padding去掉
    return _remove_padding(padding_dz, padding)
