import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from OfficialWork.SIOA.TrafficFlowForecast.SourceCode.DP_LSTM import *
#from DP_LSTM import *


# 输入数据文件， 输出每分钟总流量
def data_process(data_pd):
    time_iter = list(np.unique(data_pd['时间']))
    flow_container = []
    time_container = []
    for time_point in time_iter:
        selected_pd = data_pd.loc[data_pd['时间'] == time_point]
        flow_val = np.sum(selected_pd['上车人数']) + np.sum(selected_pd['下车人数'])
        time_container.append(time_point)
        flow_container.append(flow_val)
    return pd.DataFrame({'Time': time_container, 'Flow_Val': flow_container})


# 输入timescale，输出滚动窗口长度timescale下，timescale时间范围内的流量
def statistic_flow_scale(flow_pd, time_scale):
    flow_container = []
    time_container = []
    for index in range(flow_pd.shape[0] - time_scale + 1):
        time_container.append(flow_pd['Time'][index])
        flow_container.append(np.sum(flow_pd['Flow_Val'][index:index + time_scale]))
    res_pd = pd.DataFrame({'Time': time_container, 'Flow_sum': flow_container})
    res_pd['Time'] = pd.to_datetime(res_pd['Time'])
    return res_pd


def data_normalized(raw_df):  # 标准化数据
    data_pd = raw_df.copy()
    mm_scaler = MinMaxScaler(feature_range=(0.1, 1))
    scaled_Data = mm_scaler.fit_transform(
        np.reshape(data_pd['Flow_sum'].get_values(), (data_pd['Flow_sum'].shape[0], 1)))
    data_pd['Flow_sum_trans'] = scaled_Data
    return data_pd, mm_scaler


def get_timeSeries(normalized_pd, feature, TimeLag):  # 根据时滞创建时间序列数据
    # 输入 标准化后的完整数据、时滞TimeLag
    # 输出 时间序列数据 np.array
    sequence_length = TimeLag + 1
    result = []
    time_container = []
    for index in range(len(normalized_pd) - sequence_length + 1):
        result.append(normalized_pd[feature][index: index + sequence_length])
        time_container.append(normalized_pd['Time'][index])
    return np.array(result), time_container


def lstm_predict(data_pd, params):
    # 定义存储结果的pd
    result_pd = pd.DataFrame(columns=['TimeLags', 'RMSE', 'MAE', 'MAPE'])
    line_pd = pd.DataFrame(columns=['TimeLags', 'RMSE', 'MAE', 'MAPE'])
    # 对数据进行预处理
    # 计算每分钟值
    flow_pd = data_process(data_pd)
    # 按时间跨度进行汇总
    sum_pd = statistic_flow_scale(flow_pd, params['Timescale'])
    # 归一化
    normalized_pd, mm_scaler = data_normalized(sum_pd)
    print(r'处理数据:', normalized_pd.shape)
    # 构造时间序列数据
    TimeSeries, time_container = get_timeSeries(normalized_pd, 'Flow_sum_trans', TimeLag=params['TimeLag'])
    # 划分训练测试集
    TrainSeries = TimeSeries[:round(params['TrainTestSplit'] * TimeSeries.shape[0]), :]
    print(r'训练数据:', TrainSeries.shape)
    TestSeries = TimeSeries[round(params['TrainTestSplit'] * TimeSeries.shape[0]):, :]
    print(r'验证数据:', TestSeries.shape)
    TrainSeries_X = TrainSeries[:, :-1]
    TrainSeries_Y = TrainSeries[:, -1]
    TestSeries_X = TestSeries[:, :-1]
    TestSeries_Y = TestSeries[:, -1]
    # 张量化
    Train_X = np.reshape(TrainSeries_X, (TrainSeries_X.shape[0], 1, TrainSeries_X.shape[1]))
    Train_Y = np.reshape(TrainSeries_Y, (TrainSeries_Y.shape[0], 1, 1))
    Test_X = np.reshape(TestSeries_X, (TestSeries_X.shape[0], 1, TestSeries_X.shape[1]))
    Test_Y = np.reshape(TestSeries_Y, (TestSeries_Y.shape[0], 1, 1))
    print(r'LSTM预测')
    lstm_model = build_lstm_model(inputDim=Train_X.shape[2], lr=params['lr'], nb_hidden_cell=params['nb_hidden_cell'])
    fitted_model = model_fit(lstm_model, Train_X, Train_Y, validation_split=0.2, epochs=params['epochs'],
                             batch_size=params['batch_size'])
    RMSE, MAE, MAPE, Predict_y, True_Y = model_predict(fitted_model, Test_X, Test_Y, mm_scaler)
    line_pd['RMSE'] = [RMSE]
    line_pd['MAE'] = [MAE]
    line_pd['MAPE'] = [MAPE]
    line_pd['TimeLags'] = [params['TimeLag']]

    return fitted_model, TimeSeries, time_container, mm_scaler


# 输入x, y 画出真实值和预测值随时间分布图,默认用全部数据重新进行预测并绘制图片（mode='full'），
# mode设置为其他的值就只用测试数据进行绘图
def predict_and_plot(fitted_model, TimeSeries, time_container, mm_scaler, param, mode='full'):
    arrary_x = TimeSeries[:, :-1]
    label_y = TimeSeries[:, -1]
    if mode != 'full':
        arrary_x = TimeSeries[round(len(TimeSeries) * param['TrainTestSplit']):, :-1]
        label_y = TimeSeries[round(len(TimeSeries) * param['TrainTestSplit']):, -1]
        time_container = time_container[round(len(TimeSeries) * param['TrainTestSplit']):]

    tensor_x = np.reshape(arrary_x, (arrary_x.shape[0], 1, arrary_x.shape[1]))
    tensor_y = np.reshape(label_y, (label_y.shape[0], 1, 1))

    predict_val = fitted_model.predict(tensor_x)
    predict_val = mm_scaler.inverse_transform(predict_val.reshape(predict_val.shape[0], 1))

    true_val = mm_scaler.inverse_transform(tensor_y.reshape(tensor_y.shape[0], 1))

    rmse = evaluation_RMSE(true_val, predict_val)
    mae = evaluation_MAE(true_val, predict_val)
    mape = evaluation_MAPE(true_val, predict_val)
    print('RMSE:%s  ,MAE:%s  ,MAPE:%s ' % (rmse, mae, mape))
    # 绘图
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    mondayFormatter = DateFormatter('%H:%M')
    plt.plot(time_container, true_val, 'g-s', label='Real', color='black')
    plt.plot(time_container, predict_val, 'g-o', label='Predict-TimeScale_%s' % param['Timescale'], color='g')
    plt.xlabel(r'Time')
    plt.ylabel(r'Volumn')
    ax.xaxis.set_major_formatter(mondayFormatter)
    for rt in ax.get_xticklabels():
        rt.set_rotation(30)
    plt.legend()
    plt.savefig(param['pic_path'] + '/Predict-time_scale_%s_mode_%s' % (param['Timescale'], mode) + r'.png')
    plt.show()


if __name__ == '__main__':
    data_pd = pd.read_csv(r'/home/wells/Wells/OfficialWork/SIOA/TrafficFlowForecast/SourceData/数据.csv', encoding='GBK')
    for mode in ['full', 'Test_only']:
        for time_scale in [5, 15, 30]:
            params = {
                'Timescale': time_scale,
                'TimeLag': 5,
                'lr': 1e-4,
                'nb_hidden_cell': [80, 80],
                'epochs': 2000,
                'batch_size': 30,
                'pic_path': r'/home/wells/Wells/OfficialWork/SIOA/TrafficFlowForecast/ResultPic',
                'TrainTestSplit': 0.6}
            fitted_model, TimeSeries, time_container, mm_scaler = lstm_predict(data_pd, params)
            predict_and_plot(fitted_model, TimeSeries, time_container, mm_scaler, params, mode=mode)