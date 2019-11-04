# 拉格朗日发插补空缺值

import pandas as pd     # 导入pandas库
from scipy.interpolate import lagrange     # 导入拉格朗日函数

inputfile = u'C:\\基于深度Q学习的交通状态预测\\Data.xls'
outputfile = u'C:\\基于深度Q学习的交通状态预测\\Data.xls'

data = pd.read_excel(inputfile)
data[u'速度'][(data[u'速度'] < 20) | (data[u'速度'] > 80)] = None     # 清空异常值


def ployinterp_colum(s, n, k=5):     # 用空值前后5个数值来拟合曲线，从而预测控制
    y = s[list(range(n-k, n))+list(range(n+1, n+1-k))]     # 取值，range函数返回一个左闭右开的序列数
    y = y[y.notnull()]     # 从上一行中取出数值列表中的非空值，保证y的每行数都有数值，用于拟合
    return lagrange(y.index, list(y))(n)     # 调用拉格朗日函数，并添加索引


for i in data.colums:     # 如果i在data的列名中，data.column生成的是data的全部列名
    for j in range(len(data)):     # len(data)返回了data的长度，若长度为11，则range(11)会产生从0开始计数的整数列表
        if (data[i].isnull())[j]:     # 如果data[i][j]为空，则调用函数ployinterp_column为其插值
            data[i][j] = ployinterp_column(data[i], j)

data.to_excel(outputfile)     # 将完成插值后的data写入excel
print("拉格朗日法插值完成，插补后的文件位于:"+str(outputfile))