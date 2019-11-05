import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 常量的定义
rnn_unit = 10     # 隐层的神经元个数，区间为[4,13]
input_size = 6     # 输入层的神经元个数(日期、车速、延迟时间、行程时间、温度、下雨概率)
output_size = 1     # 输出层的神经元个数
lr = 0.06     # 最佳学习速率为[0.05,0.2]

# 导入数据
f = open('C:/基于深度Q学习的交通状态预测/内容整理/数据整理/Data.csv')
df = pd.read_csv(f)     # 读入交通数据
data = df.iloc[:, 0: 6].values     # 取第1~6列


# 获取训练集
def get_train_data(batch_size=100, time_step=15, train_begin=0, train_end=1500):
    batch_index = []
    data_train = data[train_begin:train_end]
    normalized_train_data = (data_train-np.mean(data_train, axis=0))/np.std(data_train, axis=0)     # 标准化
    train_x, train_y = [], []     # 训练集
    for i in range(len(normalized_train_data)-time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i:i+time_step, :6]
        y = normalized_train_data[i:i+time_step, 6, np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index, train_x, train_y


# 获取测试集
def get_test_data(time_step=15, test_begin=1501):
    data_test = data[test_begin:]
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    normalized_test_data = (data_test-mean)/std     # 标准化
    size = (len(normalized_test_data)+time_step-1)//time_step     # 有size个sample
    test_x, test_y = [], []
    for i in range(size-1):
        x = normalized_test_data[i*time_step:(i+1)*time_step, :6]
        y = normalized_test_data[i*time_step:(i+1)*time_step, 7]
        test_x.append(x.tolist())
        test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:, :7]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:, 7]).tolist())
    return mean, std, test_x, test_y


# 输入层、输出层权重、偏置

weights = {
    'in': tf.Variable(tf.random.normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random.normal([rnn_unit, 1]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}


# --------------定义神经网络变量-----------------
def lstm(X):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, input_size])      # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in)+b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])     # 将tensor转成3维，作为lstm cell输入
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output = tf.reshape(output_rnn, [-1, rnn_unit])     # 作为输出层的输入
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out)+b_out
    return pred, final_states


# ----------------训练模型---------------
def train_lstm(batch_size=100, time_step=15, train_begin=0, train_end=1500):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    # 训练样本中第1-1500个样本，每次取15个，共取100次
    batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)
    print(np.array(train_x).shape)
    print(batch_index)
    pred, _ = lstm(X)
    # 损失函数
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1])-tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 重复训练100次
        for i in range(100):     # 每次进行训练的时候，每个batch训练batch_size个样本
            for step in range(len(batch_index)-1):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[batch_index[step]:batch_index[step+1]],
                                                                 Y: train_y[batch_index[step]:batch_index[step+1]]})
            print(i, loss_)
            if i % 100 == 0:
                print("保存模型:", saver.save(sess, 'model/stock2.model', global_step=i))


train_lstm()


# ------------预测模型-------------
def prediction(time_step=15):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    mean, std, test_x, test_y = get_test_data(time_step)
    pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:     # 参数恢复
        module_file = tf.train.latest_checkpoint('model')
        saver.restore(sess, module_file)
        test_predict = []
        for step in range(len(test_x)-1):
            prob = sess.run(pred, feed_dict={X: [test_x[step]]})
            predict = prob.reshape((-1))
            test_predict.extend(predict)
        test_y = np.array(test_y)*std[6]+mean[6]
        test_predict = np.array(test_predict)*std[6]+mean[6]
        acc = np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])     # 偏差
        # 以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b')
        plt.plot(list(range(len(test_y))), test_y, color='r')
        plt.show()


prediction()
