import numpy as np
import pandas as pd
import datetime
from keras.callbacks import Callback as cb
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from matplotlib import pyplot as plt

'''LSTM建模模块'''
def build_LSTM_Model(inputDim = 10,lr=0.001,nb_hidden_cell=[80]):
    '''
    Parameters
        inputDim: 训练数据的维度（不含预测值）
    Reutrn
        return: model 用于训练，并将图形保存为D:\WORK__wells\PROGRAM_3\Model pic\LTSMmodel.png
    '''
    from keras.optimizers import RMSprop,Adam
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers.recurrent import LSTM
    from keras.layers.recurrent import GRU
    from keras.layers.core import Dropout
    model = Sequential()

    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    rmsprop = RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0001)  # 自适应学习速率。decay学习速率衰减值
    model.add(LSTM(nb_hidden_cell[0],input_shape=(1,inputDim),use_bias=True,activation='tanh',name='layer_0',return_sequences=True))
#     model.add(LSTM(80,input_shape=(None,inputDim),name='layer_0',return_sequences=True))
#     model.add(GRU(nb_hidden_cell[0], activation='relu',return_sequences=True))
#     model.add(Dropout(0.2))
    for i in range(1,len(nb_hidden_cell)):
        model.add(LSTM(nb_hidden_cell[i],activation='tanh',use_bias=True, name='layer_%s'%i,return_sequences=True))
#         model.add(Dropout(0.1))
    model.add(Dense(1,use_bias=True,activation='linear',name='last'))
    # 调整优化器
    model.compile(optimizer=adam,loss='mse',metrics=['mae','mape'])
    return model

def build_BP_Model(inputDim=10,lr=0.001,nb_hidden_cell=[80]):
    from keras.optimizers import RMSprop,Adam
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers.core import Dropout
    model = Sequential()
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    rmsprop = RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0001)  # 自适应学习速率。decay学习速率衰减值
    model.add(Dense(nb_hidden_cell[0],input_dim=inputDim,use_bias=True,activation='relu',name='first'))
    for i in range(1,len(nb_hidden_cell)):
        model.add(Dense(nb_hidden_cell[i], use_bias=True,activation='tanh', name='layer_%s' % i))
        model.add(Dropout(0.2))
    model.add(Dense(1,use_bias=True,activation='linear',name='last'))
    model.compile(optimizer=adam,loss='mse',metrics=['mae','mape'])
    return model

def model_fit(model,x_train,y_trian,validation_split=0.3,saveFile='E:\SIOA\Program\PersonalProfit\DowJonesPredict\SourceCode\ModelFile',epochs=10,batch_size=30):
    from keras.callbacks import ModelCheckpoint
    #创建一个实例LossHistory
    history = LossHistory()
#     early_stopping = EarlyStopping(monitor='val_loss', patience=50)
    checkpointer = ModelCheckpoint(filepath=saveFile + '\checkpoint.hdf5',
                                   monitor = 'val_mean_absolute_percentage_error',save_best_only=True)
    history_=model.fit(x_train,y_trian,
                       validation_split=validation_split,
                       epochs=epochs,batch_size=batch_size,
                       callbacks = [history,checkpointer],shuffle=False) # shuffle=False因为是时间序列，不能打乱顺序
    model.save(saveFile + '\Model.h5')
    history.loss_plot('epoch')
    return model

def evaluation_RMSE(y_test, predict_array):
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(y_test, predict_array))

def evaluation_MAE(y_test, predict_array):
    from sklearn.metrics import mean_absolute_error
    return mean_absolute_error(y_test, predict_array)


def evaluation_MAPE(y_true, y_pred): 
    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def model_predict(model,x_test,y_test,scaler,title='predict',save=r'E:\SIOA\Program\PersonalProfit\DowJonesPredict\SourceCode\PicSave',Y_reverse=1):
    predict_array = model.predict(x_test)
    
    predict_array = scaler.inverse_transform(predict_array.reshape(predict_array.shape[0],1))
    if Y_reverse==1:
        y_test = scaler.inverse_transform(y_test.reshape(y_test.shape[0],1))
    y_test = y_test.reshape(y_test.shape[0],1)
    RMSE = evaluation_RMSE(y_test,predict_array)
    MAE = evaluation_MAE(y_test,predict_array)
    MAPE = evaluation_MAPE(y_test,predict_array)


    plt.plot(y_test, 'b', label='Ground Truth')
    plt.plot(predict_array, 'r', label='Prediction')
    plt.title('predict')
#     plt.xlabel('Hours')
    plt.ylabel('Exchange Rate')
    # plt.title(title+str(nowTime_))
#     plt.xlabel(kwargs)
    # plt.ylabel('RMSE:%s  ,MAE:%s  ,MAPE:%s '%(RMSE,MAE,MAPE))
    plt.legend(shadow=True)
#     saveFig = r'D:\WORK__wells\PROGRAM_3\Model pic\Predict.png'
    plt.savefig(save + '\Predict.png', dpi=2400)
    plt.show()
    print('RMSE:%s  ,MAE:%s  ,MAPE:%s '%(RMSE,MAE,MAPE))
    return RMSE,MAE,MAPE,predict_array,y_test

def model_predictVal(model,x_test,scaler,title='predict',save=r'E:\SIOA\Program\PersonalProfit\DowJonesPredict\SourceCode\PicSave',**kwargs):
    predict_array = model.predict(x_test)
#     predict_array = scaler.inverse_transform(predict_array.reshape(predict_array.shape[0],1))
    return predict_array


class LossHistory(cb):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_mae = {'batch': [], 'epoch': []}
        self.val_mape = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        
        self.val_mae['batch'].append(logs.get('val_mae'))
        self.val_mape['batch'].append(logs.get('val_mape'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        
        self.val_mae['epoch'].append(logs.get('val_mae'))
        self.val_mape['epoch'].append(logs.get('val_mape'))

    def loss_plot(self, loss_type = 'epoch'):
        iters = range(len(self.losses[loss_type]))
        #创建一个图
        plt.figure()
        # acc
        plt.plot(iters, self.losses[loss_type], 'r', label='loss')#plt.plot(x,y)，这个将数据画成曲线
        # loss
#         plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
            plt.plot(iters, self.val_mae[loss_type], 'b', label='val_mae')
            plt.plot(iters, self.val_mape[loss_type], 'b', label='val_mape')
        plt.grid(True)#设置网格形式
        plt.xlabel(loss_type)
        plt.ylabel('loss')#给x，y轴加注释
        plt.legend(loc="upper right")#设置图例显示位置
        plt.show()





