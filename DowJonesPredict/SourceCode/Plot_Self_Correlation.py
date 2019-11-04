
# coding: utf-8

# 计算相关系数
import numpy as np
from matplotlib import pyplot as plt

def get_auto_corr(timeSeries,k):
    '''
    Descr:输入：时间序列timeSeries，滞后阶数k
            输出：时间序列timeSeries的k阶自相关系数
        l：序列timeSeries的长度
        timeSeries1，timeSeries2:拆分序列1，拆分序列2
        timeSeries_mean:序列timeSeries的均值
        timeSeries_var:序列timeSeries的每一项减去均值的平方的和
        
    '''
    l = len(timeSeries)
    #取出要计算的两个数组
    timeSeries1 = timeSeries[0:l-k]
    timeSeries2 = timeSeries[k:]
    
    timeSeries_mean = timeSeries.mean()
    timeSeries_var = np.array([i**2 for i in timeSeries-timeSeries_mean]).sum()
#     print(timeSeries_var)
    auto_corr = 0
    for i in range(l-k):
#         print(timeSeries2[i])
        temp = (timeSeries1[i]-timeSeries_mean)*(timeSeries2[i]-timeSeries_mean)/timeSeries_var
        auto_corr = auto_corr + temp  
    return auto_corr
 
#画出各阶自相关系数的图
def plot_auto_corr(timeSeries,k):
    '''
    Descr:需要计算自相关函数get_auto_corr(timeSeries,k)
            输入时间序列timeSeries和想绘制的阶数k，k不能超过timeSeries的长度
            输出：k阶自相关系数图，用于判断平稳性
    '''
    timeLat_Coval = []
    for i in range(1,k+1):
        timeLat_Coval.append(get_auto_corr(timeSeries,i))
    
    plt.plot(range(1,len(timeLat_Coval)+1),timeLat_Coval,linewidth=4)
    plt.plot(range(1,len(timeLat_Coval)+1), [0.5 for i in range(len(timeLat_Coval))],linestyle=':')
    plt.xlim(1,k)
    plt.xlabel(r'Time lag ')
    plt.ylabel(r'Auto-correlation coeffcients')
    
    SavePath = r'D:\WORK__wells\PROGRAM_3\pic\\'
    plt.savefig(SavePath +'Time lag'+ str(k)+ r'.jpg',dpi=2400)
    return timeLat_Coval

#画出各阶自相关系数的图____多站点
def plot_auto_corr_multi_station(timeSerieslist,k):
    '''
    Descr:需要计算自相关函数get_auto_corr(timeSeries,k)
            输入时间序列timeSeries和想绘制的阶数k，k不能超过timeSeries的长度
            输出：k阶自相关系数图，用于判断平稳性
    '''
    
    timeSeries = timeSerieslist[0]  # 5 号站点
    timeLat_Coval = []
    for i in range(1,k+1):
        timeLat_Coval.append(get_auto_corr(timeSeries,i))
    plt.plot(range(1,len(timeLat_Coval)+1),timeLat_Coval,linewidth=4,label='#5 highway station')
    plt.legend()
    
    timeSeries = timeSerieslist[1]  # 6055 号站点
    timeLat_Coval = []
    for i in range(1,k+1):
        timeLat_Coval.append(get_auto_corr(timeSeries,i))
    plt.plot(range(1,len(timeLat_Coval)+1),timeLat_Coval,linewidth=4,label='#605 highway station')
    plt.legend()
    plt.plot(range(1,len(timeLat_Coval)+1), [0.5 for i in range(len(timeLat_Coval))],linestyle=':')
        
    plt.xlim(1,k)
    plt.xlabel(r'Time lag ')
    plt.ylabel(r'Auto-correlation coeffcients')
    
    SavePath = r'D:\WORK__wells\PROGRAM_3\pic\\'
    plt.savefig(SavePath +'Time lag'+ str(k)+ r'.jpg',dpi=2400)
    return timeLat_Coval