# -*- coding:utf-8 -*-
"""
加载MNIST数据集
"""
import numpy as np
import struct
import os
from tools import oneHot


def _decode_idx3_ubyte(idx3_file):
    # 二进制形式打开文件（idx3：三维文件，包括图片横纵以及数量）
    bin_data = open(idx3_file, 'rb').read()
    # offset表示偏移位置
    offset = 0
    # iiii表示4个int，>表示大端规则，是存储规则
    fmt_header = '>iiii'
    # 解包的时候得到4个int，给4个变量赋值
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)

    image_size = num_rows * num_cols
    # header表示的是文件头部，是上面获得的4个int信息，因此进行偏移，以读取后面的内容；此处offset为16
    offset += struct.calcsize(fmt_header)
    # 图片部分的格式，相当于分段的标记，B表示字节，相当于提取784字节
    fmt_image = '>' + str(image_size) + 'B'
    # 使用矩阵进行存储
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        # 使用unpack_from提取得到的不再是元组，而是一个值；需根据偏移位置决定往后提取的内容
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        # 偏移位置发生改变
        offset += struct.calcsize(fmt_image)
    return images


def _decode_idx1_ubyte(idx1_file):
    bin_data = open(idx1_file, 'rb').read()

    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)

    offset += struct.calcsize(fmt_header)
    # 相当于>1B
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def _load_train_images(file):
    """
    获取（N，28，28）的数据，然后转换shape
    """
    train_images = _decode_idx3_ubyte(file)
    r = train_images.shape[0]
    return train_images.reshape((r, 1, 28, 28))


def _load_train_labels(file, onehot=True):
    if onehot:
        return oneHot(_decode_idx1_ubyte(file))
    else:
        return _decode_idx1_ubyte(file)


def _load_test_images(file):
    return _decode_idx3_ubyte(file).reshape((_decode_idx3_ubyte(file).shape[0], 1, 28, 28))


def _load_test_labels(file, onehot=True):
    if onehot:
        return oneHot(_decode_idx1_ubyte(file))
    else:
        return _decode_idx1_ubyte(file)


def train_load(mnist_path=r"E:\SIOA\Program\PersonalProfit\CNN\MNIST", onehot=True):
    """
    :param mnist_path: 4个解压过的文件所在文件夹路径
    :param onehot: 是否OneHot
    :return: 训练集中的images和labels对象
    """
    train_images_path = os.path.join(mnist_path, "train-images.idx3-ubyte")
    train_labels_path = os.path.join(mnist_path, "train-labels.idx1-ubyte")
    tr_i = _load_train_images(file=train_images_path)
    tr_l = _load_train_labels(file=train_labels_path, onehot=onehot)
    return tr_i, tr_l


def test_load(mnist_path="E:\SIOA\Program\PersonalProfit\CNN\MNIST", onehot=True):
    """
    :param mnist_path: 4个解压过的文件所在文件夹路径
    :param onehot: 是否OneHot
    :return: 测试集中的images和labels对象
    """
    test_images_path = os.path.join(mnist_path, "t10k-images.idx3-ubyte")
    test_labels_path = os.path.join(mnist_path, "t10k-labels.idx1-ubyte")
    te_i = _load_test_images(file=test_images_path)
    te_l = _load_test_labels(file=test_labels_path, onehot=onehot)
    return te_i, te_l
