import os
import cv2
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from OfficialWork.SIOA.WheatClassification.SourceCode.DataInitialize import DataInitialize
#from DataInitialize import DataInitialize


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(1.0, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride_length=1,padding='VALID'):
    return tf.nn.conv2d(x, W, strides=[1, stride_length, stride_length, 1], padding=padding)


def max_pool_2x2(x,pool_size=2,stride_length=2,padding='VALID'):
    return tf.nn.max_pool(x, ksize=[1, pool_size, pool_size, 1],
                          strides=[1, stride_length, stride_length, 1], padding=padding)


def cnn_build(x_image=tf.placeholder('float', shape=[None, 640,128,1]),input_shape=(640,128)):
    # 第一层卷积
    filter_conv1 = weight_variable([9, 9, 1, 16])
    bias_conv1 = bias_variable([16])
    h_conv1 = tf.nn.relu(conv2d(x_image, filter_conv1) + bias_conv1)
    # 池化
    h_pool1 = max_pool_2x2(h_conv1) # (?, 316, 60, 16)
    # 第二层卷积
    filter_conv2 = weight_variable([9, 9, 16, 32])
    bias_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, filter_conv2) + bias_conv2) # (?,307, 52, 32)
    # 池化
    h_pool2 = max_pool_2x2(h_conv2) # (?, 154, 26, 32)
    # 第三层卷积
    filter_conv3 = weight_variable([5, 5, 32, 64])
    bias_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, filter_conv3) + bias_conv3) # (?, 150, 22, 64)
    # 池化
    h_pool3 = max_pool_2x2(h_conv3)  #  (?, 75, 11, 64)
    # 第四层卷积
    filter_conv4 = weight_variable([3, 3, 64, 128])
    bias_conv4 = bias_variable([128])
    h_conv4 = tf.nn.relu(conv2d(h_pool3, filter_conv4) + bias_conv4)  # (?, 73, 9, 128)
    # 将第四层卷积层舒展
    h_conv4_flat = tf.reshape(h_conv4, [-1, 73*9*128])
    # 全连接层1
    W_fc1 = weight_variable([73*9*128, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv4_flat, W_fc1) + b_fc1)  # (?, 1024)
    # 全连接层2
    W_fc2 = weight_variable([1024, 512])
    b_fc2 = bias_variable([512])
    h_fc2=tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)  # (?, 512)
    return h_fc2


def train_model(data_initializer, input_shape=(640, 128)):
    sess = tf.InteractiveSession()
    # 三通道image
    x_image_channel1 = tf.placeholder('float', shape=[None, input_shape[0], input_shape[1], 1])
    x_image_channel2 = tf.placeholder('float', shape=[None, input_shape[0], input_shape[1], 1])
    x_image_channel3 = tf.placeholder('float', shape=[None, input_shape[0], input_shape[1], 1])
    # 真实值
    y_true = tf.placeholder('float', shape=[None, 1])
    # 三通道全连接层2输出
    h_fc_channe1 = cnn_build(x_image_channel1, input_shape)
    h_fc_channe2 = cnn_build(x_image_channel2, input_shape)
    h_fc_channe3 = cnn_build(x_image_channel3, input_shape)
    # 将三通道的全连接层输出进行fusion
    concat_1 = tf.concat([h_fc_channe1, h_fc_channe2], axis=1)  # [h_fc_channe1, h_fc_channe2]
    concat_2 = tf.concat([concat_1, h_fc_channe3], axis=1)
    fc_final = tf.reshape(concat_2, [-1, 512 * 3])
    # dropout 层
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(fc_final, keep_prob=keep_prob)
    # softmax 输出层y_conv
    W_fc3 = weight_variable([512 * 3, 1])
    b_fc3 = bias_variable([1])
    logits = tf.matmul(h_fc1_drop, W_fc3) + b_fc3
    y_conv = tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc3) + b_fc3)
    # 计算交叉熵
    # tf.log(tf.clip_by_value(tf.sigmoid(self.scores),1e-8,1.0)
    # cross_entropy = -tf.reduce_sum(y_true*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=logits))

    # loss = tf.reduce_mean(tf.clip_by_value(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.arg_max(y_true,1),logits=logits),1e-8,1.0))
    # cross_entropy = -tf.reduce_sum(y_true*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)
    # correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_true,1))
    correct_prediction = tf.equal(y_conv, y_true)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # 训练模型
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        data_initializer.mode = 'train'
        train_batch = data_initializer.data_generator(16)
        image = train_batch[0]
        label = train_batch[1]
        feed_dict = {x_image_channel1: np.reshape(image[:, 0, :, :], [-1, input_shape[0], input_shape[1], 1]),
                     x_image_channel2: np.reshape(image[:, 1, :, :], [-1, input_shape[0], input_shape[1], 1]),
                     x_image_channel3: np.reshape(image[:, 2, :, :], [-1, input_shape[0], input_shape[1], 1]),
                     y_true: label,
                     keep_prob: 1
                     }
        train_step.run(feed_dict=feed_dict)
        entropy = cross_entropy.eval(feed_dict=feed_dict)
        train_accuracy = accuracy.eval(feed_dict=feed_dict)
        # print(y_conv.eval(feed_dict=feed_dict), label)
        # print("Step:%d, TrainingAccuracy:%g,cross_entropy:%.3f"%(i, train_accuracy, entropy))
        print("Step:%d,cross_entropy:%.3f" % (i, entropy))
        data_initializer.mode = 'test'
        test_batch = data_initializer.data_generator(40)
        image = test_batch[0]
        label = test_batch[1]
        feed_dict = {x_image_channel1: np.reshape(image[:, 0, :, :], [-1, input_shape[0], input_shape[1], 1]),
                     x_image_channel2: np.reshape(image[:, 1, :, :], [-1, input_shape[0], input_shape[1], 1]),
                     x_image_channel3: np.reshape(image[:, 2, :, :], [-1, input_shape[0], input_shape[1], 1]),
                     y_true: label,
                     keep_prob: 1
                     }
        print("test accuracy %g" % accuracy.eval(feed_dict=feed_dict))
        # print(y_conv.eval(feed_dict=feed_dict),label)


if __name__ == '__main__':
    data_initializer = DataInitialize(
        root_path=r'E:\OfficialWorkRemote\OfficialWork\SIOA\WheatClassification\SourceData',
        train_test_rate=0.6, reshape_size=(128, 640), mode='train')
    train_model(data_initializer)