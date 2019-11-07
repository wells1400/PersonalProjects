import numpy as np
import pandas as pd
import random
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import LSTM

from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt

np.random.seed(1)


# 数据处理
class DataProcess:
    def __init__(self, data_path):
        self.data_path = data_path
        # 分割数据
        self.data_split()

    def data_split(self):
        data_pd = pd.read_csv(self.data_path, encoding='gbk')
        if not hasattr(self, 'data_pd'):
            self.data_pd = data_pd
        data_pd = data_pd.iloc[:, 1:]
        data_pd = data_pd.rename({'实际时间': 'actual_time', '速度': 'speed',
                                  '延迟时间': 'delay_time', '行程时间': 'road_time',
                                  '气温': 'temperature', '降水概率': 'rain_prob'}, axis=1)
        if not hasattr(self, 'trian_pd'):
            self.train_pd = data_pd.loc[:1500, :]
        if not hasattr(self, 'test_pd'):
            self.test_pd = data_pd.loc[1500:, :]


# 交通环境模块
class TrafficEnvironment(DataProcess):
    def __init__(self, data_path):
        DataProcess.__init__(self, data_path=data_path)
        self.reset_mode(mode='train')

    # 环境模式
    def reset_mode(self, mode):
        if not hasattr(self, 'mode'):
            self.mode = mode
        self.mode = mode

    # 环境交互函数
    # 输入状态s、行为，输出s,a,reward和s_,如果找不到满足条件的reward和s_则返回空[]
    def env_backward(self, s, a):
        check_s_ = self.check_action(s, a)
        if len(check_s_) == 0:
            return []
        s_ = random.choice(check_s_)
        reward = s[1] - s_[1]
        return s, a, reward, s_

    # 检查行为是否合法,合法则返回[s_1,s_2...],不合法则返回[]
    def check_action(self, s, a):
        data_pd = self.train_pd if self.mode == 'trian' else self.test_pd
        action_feature = [0 for _ in range(data_pd.shape[1])]

        decoded_str = bin(a).replace('0b', '')
        for val in decoded_str:
            val_int = int(val)
            action_feature.append(val_int)
        # 表征行为的向量
        action_vector = action_feature[-self.train_pd.shape[1]:]
        col_labels = self.train_pd.columns.tolist()
        # 按照条件逐步筛选s_
        selected_pd = data_pd
        for filter_index in range(len(action_vector)):
            if action_vector[filter_index] == 1:
                selected_pd = selected_pd.loc[selected_pd[col_labels[filter_index]] >= s[filter_index]]
            else:
                selected_pd = selected_pd.loc[selected_pd[col_labels[filter_index]] < s[filter_index]]
            if selected_pd.shape[0] == 0:
                return []
        s_next_container = selected_pd.get_values().tolist()
        return s_next_container


class DeepRecurrentQNetwork:
    def __init__(self, n_features, learning_rate=0.01, reward_decay=0.9,
                 e_greedy=0.9, replace_target_iter=300, memory_size=500,
                 batch_size=32, e_greedy_increment=None):
        self.n_features = n_features  # 表示状态s的特征
        self.n_actions = 2 ** self.n_features  # 可选行为的个数取决于s的特征长度

        self.lr = learning_rate  # 学习率
        self.gamma = reward_decay  # reward衰减的系数
        self.epsilon_max = e_greedy  # epsilon最大值
        self.replace_target_iter = replace_target_iter  # 每嗝多少次训练就轮换一下网络的参数
        self.memory_size = memory_size  # 记忆库最大容量
        self.batch_size = batch_size  # 训练的batchsize
        self.epsilon_increment = e_greedy_increment  # epsilon增加量，随着训练的迭代，逐步减少探索性
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self.build_net()

        self.cost_his = []

    # 构建记忆库
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # 更新记忆，替换
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    # 选择行为
    def choose_action(self, s):
        valid_actions = []
        for action in range(self.n_actions):
            env_backward = env.check_action(s, action)
            if len(env_backward) != 0:
                valid_actions.append(action)
        # 任何一个状态s不可能没有行为可以选
        rand_point = np.random.uniform()
        if rand_point < self.epsilon:
            s = np.reshape(s, (1, -1, len(s)))
            # epsilon的概率从valid_actions中选择最优的那个
            actions_value = self.model_eval_net.predict(s)
            actions_value = np.reshape(actions_value, (-1, 1))

            action = valid_actions[0]
            for action_index in valid_actions:
                action = action_index if actions_value[action_index] > actions_value[action] else action
            return action
        else:
            # 1-epsilon的概率valid_actions中随机选择一个行为
            action = np.random.choice(valid_actions)
        return action

    # 搭建神经网络
    def build_net(self):
        adam = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        # ----------eval_net------------
        model_eval_net = Sequential()
        model_eval_net.add(
            LSTM(128, input_shape=(1, self.n_features), return_sequences=True, activation='tanh', use_bias=True,
                 name='input_layer'))
        model_eval_net.add(LSTM(256, return_sequences=True, activation='tanh', use_bias=True, name='hidden_layer'))
        model_eval_net.add(
            LSTM(2 ** self.n_features, return_sequences=False, activation='sigmoid', use_bias=True, name='out_layer'))
        model_eval_net.compile(optimizer=adam, loss='mse', metrics=['mae', 'mape'])

        # ----------target_net------------
        model_target_net = Sequential()
        model_target_net.add(
            LSTM(128, input_shape=(1, self.n_features), return_sequences=True, activation='tanh', use_bias=True,
                 name='input_layer'))
        model_target_net.add(LSTM(256, return_sequences=True, activation='tanh', use_bias=True, name='hidden_layer'))
        model_target_net.add(
            LSTM(2 ** self.n_features, return_sequences=False, activation='sigmoid', use_bias=True, name='out_layer'))
        model_target_net.compile(optimizer=adam, loss='mse', metrics=['mae', 'mape'])

        print(model_eval_net.summary())
        plot_model(model_eval_net, to_file='model_eval_net.png', show_shapes=True)

        print(model_target_net.summary())
        plot_model(model_target_net, to_file='model_target_net.png', show_shapes=True)

        if not hasattr(self, 'model_eval_net'):
            self.model_eval_net = model_eval_net
        if not hasattr(self, 'model_target_net'):
            self.model_target_net = model_target_net

    # 模型参数替换
    def target_replace_op(self):
        tmp_params = self.model_eval_net.get_weights()
        self.model_target_net.set_weights(tmp_params)

    # 训练提升网络
    def learn(self):
        if self.learn_step_counter != 0 and self.learn_step_counter % self.replace_target_iter == 0:
            self.target_replace_op()
            print('target_params_replaced')
        # 从记忆库中抽样batch_size
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.model_eval_net.predict(
            batch_memory[:, -self.n_features:].reshape(-1, 1, self.n_features)), \
                         self.model_target_net.predict(
                             batch_memory[:, :self.n_features].reshape(-1, 1, self.n_features))

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        self.model_eval_net.fit(batch_memory[:, :self.n_features].reshape(-1, 1, self.n_features), q_target, epochs=20,
                                verbose=1)

        score = self.model_eval_net.evaluate(batch_memory[:, :self.n_features].reshape(-1, 1, self.n_features),
                                             q_target)
        self.cost_his.append(score)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    # 绘制损失曲线
    def plot_history(self):
        loss_his_pd = pd.DataFrame(self.cost_his, columns=['loss', 'mae', 'mape'])
        plt.plot(np.arange(loss_his_pd.shape[0]), loss_his_pd['loss'])
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.savefig('loss history.png')
        plt.show()


# 网络训练主程序
def train_main(env, rl_network, max_episode, net_work_train_epoch_each=200):
    step = 0
    for episode in range(max_episode):
        print('episode:%s' % episode)
        # 初始化状态s
        s = random.choice(env.train_pd.values.tolist())
        while True:
            # 选择行为
            a = rl_network.choose_action(s)

            # 状态确定，行为选定后环境给出反馈
            s, a, reward, s_ = env.env_backward(s, a)
            # 将经验存储于记忆池
            rl_network.store_transition(s, a, reward, s_)

            # 当池中有超过batch_size个经验后，每隔五个step训练一次
            if (step > rl_network.batch_size) and (step % 5 == 0):
                rl_network.learn()
            # 更新状态
            s = s_

            # 查看网络提升状态，每提升20次就重新初始化状态进行下一轮episode
            if rl_network.learn_step_counter != 0 and rl_network.learn_step_counter % net_work_train_epoch_each == 0:
                break
            step += 1
    # 绘制损失曲线
    rl_network.plot_history()


# 用测试数据集预测计算q_value
def traffic_predict(env, rl_network):
    test_pd = env.test_pd
    test_pd = test_pd.reset_index(drop=True)

    test_samples = test_pd.values.reshape(-1, 1, rl_network.n_features)

    predict_value = rl_network.model_eval_net.predict(test_samples)
    values = predict_value.max(axis=1)
    time = pd.to_datetime(env.data_pd.loc[1500:, '实际时间'])
    plt.plot(time, values)
    plt.xticks(rotation=45)
    plt.ylabel('Q_value')
    plt.xlabel('Time')

    plt.savefig('Q_value.png')
    plt.show()


if __name__ == '__main__':
    data_path = r'E:\PersonalProjects\ReinforcementLearning\DRQN_traffic\SourceData\TrafficData.csv'

    env = TrafficEnvironment(data_path)

    rl_network = DeepRecurrentQNetwork(n_features=5, learning_rate=1e-4, reward_decay=0.9, e_greedy=0.9,
                                       replace_target_iter=5, memory_size=4000, batch_size=520)
    # 用训练数据集进行网络提升
    train_main(env, rl_network, max_episode=5, net_work_train_epoch_each=20)
    # 用测试数据集进行实验计算Qvalue
    traffic_predict(env, rl_network)
