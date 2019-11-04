import matplotlib.pyplot as plt
import numpy as np
import copy
import math
import random

import os

def show_result(MaxIter, bestsofar):
    plt.plot(range(MaxIter), bestsofar, '-b')
    plt.title('The best of process in de')
    plt.xlabel('iter')
    plt.ylabel('fitness')
    plt.show()

def print_result(reala, solution):
    print('the result is showing...')
    print('best_fitness for min problem:', reala[0][0])
    print('best solution for result:', solution)
    print('best solution detail for result:', reala[1])

# 传入种群大小，任务数， 机器数， 成本矩阵、资源消耗矩阵， 资源量向量， 种群解矩阵
def fitness_func(pop, NumTask, NumMac, CostMoney, CostSource, Source, gene):  # 计算目标值
    ###
    ##改变部分7：多传入了一个pop值
    ###
    buf= []
    buf.append(pop)  # 种群大小
    buf.append(NumTask)  # 任务数量  buf=[50,5]
    '''buf列表（种群大小和任务数量）'''

    fall = []
    detailall = []
    for i in range(buf[0]):  # 对每一个种群的索引i
        f = 0
        detail = np.random.randint(-1, 0, size=[3, NumTask])  # 一个3*NumTask 的矩阵，值为-1说明

        buf1 = [CostMoney[gene[i, j], j] for j in range(buf[1])]  # 存放成本值  j是任务的索引，这里是获取当前i果蝇所处位置(方案)的成本向量buf1
        buf2 = np.argsort(buf1)  # 获取buf1从小到大成本排序后的索引
        buf3 = copy.deepcopy(Source[0][:])  # 资源量向量
        for j in range(buf[1]):  # 成本从小到大遍历任务 buf2[j]是任务索引
            if buf3[gene[i, buf2[j]]]-CostSource[gene[i, buf2[j]], buf2[j]] >= 0:
                # 从成本消耗小到大算，如果这个任务在方案中安排的机器资源量大于这个机器处理这任务所需的资源消耗
                # 那么detail[0][任务索引]代表当前i果蝇代表的方案中这个任务应分配这个机器的索引
                detail[0, buf2[j]] = gene[i, buf2[j]]
                # 那么detail[1][任务索引]代表当前i果蝇代表的方案中这个任务分配给这个机器所生成的资源损耗
                detail[1, buf2[j]] = CostSource[gene[i, buf2[j]], buf2[j]]
                # 那么detail[1][任务索引]代表当前i果蝇代表的方案中这个任务分配给这个机器所生成的成本
                detail[2, buf2[j]] = CostMoney[gene[i, buf2[j]], buf2[j]]
                # 那么buf3gene[i, 任务索引]代表当前i果蝇代表的方案中这个任务消耗分配给这个机器后，资源量向量需要更新
                buf3[gene[i, buf2[j]]] = buf3[gene[i, buf2[j]]] - CostSource[gene[i, buf2[j]], buf2[j]]
        buf4 = list(detail[0])
        buf4 = list([idx for (idx, val) in enumerate(buf4) if val == -1])  # 还没有被安排的任务索引列表

        if len(buf4) >= 1:
            for j in buf4:  # 对每一个还未安排的任务的索引j
                # 对于还没被安排的任务，在最新的资源向量中查找资源量大于（果蝇i代表的原方案中处理这个任务的那个机器所需要的成本）的机器索引   XXXXXX
                buf5 = list([idx for (idx, val) in enumerate(buf3) if val >= CostSource[gene[i, j], j]])
                # 对于还没被安排的任务，记录与buf5对应的处理这些任务所需要的成本
                buf6 = list([CostMoney[idx, j] for (idx, val) in enumerate(buf3) if val >= CostSource[gene[i, j], j]])
                if len(buf5) >= 1:
                    buf7 = np.where(buf6 == min(buf6))[0]  # ????? 猜测是要找到buf6中能最小成本处理任务j的机器索引列表
                    detail[0, j] = buf5[buf7[0]] # 能最小成本处理任务j的机器索引
                    detail[1, j] = CostSource[detail[0, j], j]  # 资源消耗
                    detail[2, j] = CostMoney[detail[0, j], j] # 成本消耗
                    buf3[detail[0, j]] = buf3[detail[0, j]] - CostSource[detail[0, j], j]  # 更新资源量向量
                else:
                    print('无解')
                    f = 100000000
                    return fall, detailall
        f = sum(detail[2, :])  # 成本消耗总量
        fall.append(f)
        detailall.append(detail)
    # 返回的每个个体方案的成本消耗总量fall（向量）， detailall每个个体方案的详细信息[['任务分配机器索引','对应资源消耗'，'对应成本消耗'],:]
    return fall, detailall


def main(NumTask, NumMac, CostMoney, CostSource, Source):
    ###
    ##改变部分1：初始化数据的改动和增加
    ###
    # 随机生成数据
    #NumTask : int = 5  # 需要处理任务的数量
    #NumMac: int = 5  # 可以处理任务的机器数量
    #CostMoney = np.random.randint(1, 11, size=[NumMac, NumTask])  # 每个机器对应每个任务所耗费的成本
    #CostSource = np.random.randint(10, 100, size=[NumMac, NumTask])  # 每个机器对应每个任务所耗费的资源
    #Source = np.random.randint(100, 1000, size=[1, NumMac])  # 每个机器对应每个任务所耗费的资源
    # 算法参数
    MaxIter: int = 500  # 迭代次数
    pop : int = 50  # 种群大小
    FC : int = 0.3  # 变异概率
    FM : int = 0.7  # 交叉概率
    # 变量的取值范围
    bound = np.random.randint(1, 2, size=[2, NumTask])
    bound[0] = bound[0] * (NumMac - 1)  # 上边界，每个变量的上边界可以不同
    bound[1] = bound[1] * 0  # 下边界，每个变量的下边界可以不同
    # 生成初始代
    location = np.random.randint(0, NumMac, size=[pop, NumTask])


    [fall, rdetail] = fitness_func(pop, NumTask, NumMac, CostMoney, CostSource, Source, location)


    f_best = min(fall)  # 目前种群中成本最低的个体
    buf1 = np.where(fall == f_best)[0]
    buf1 = buf1[0]  # 种群中成本最低个体的索引
    solution = copy.deepcopy(location[buf1, :])  # 当前最佳方案的索引映射在原种群中的方案（并不一定是detail中最优的方案） XXXXXX
    detail = copy.deepcopy((rdetail[buf1]))  # 当前最佳方案的细节信息，'任务分配机器索引','对应资源消耗'，'对应成本消耗'

    ###
    ##改变部分2：两个进行变异、交叉时需要使用的数组
    ###
    locationnew = copy.deepcopy(location)  # 种群方案拷贝1
    locationnew1 = copy.deepcopy(location)  # 种群方案拷贝2
    bestsofar = []  # 最佳值的变化

    ###
    ##改变部分3：迭代逻辑
    ###
    for itera in range(MaxIter):  # 运行 代数
        print(itera)
        rf = FC*math.pow(2, math.exp(1-MaxIter/(1+MaxIter-itera)))  # 一不好玩的数学公式吧

        ###
        ## 改变部分4：变异--随机选取任意一个方案，进行位置挪动
        ###
        # 变异
        for i in range(pop): # 对一个个体
            buf = [j for j in range(pop)]  # 个体索引列表
            buf.remove(buf[i])  # 删除当前个体的索引
            buf1 = np.random.rand(1, pop-1)  # 存放49个随机浮点数值（范围0-1）
            buf2 = np.argsort(buf1)  # 获取正向排序后的下标
            # 所谓buf[buf2[0,0]]或[0,1]就是一个随机元素
            locationnew[i] = location[buf[buf2[0, 0]]] + rf*(location[buf[buf2[0, 1]]]-location[buf[buf2[0, 2]]])
            locationnew[i] = locationnew[i].astype(np.int32)  # 对随机选取的行进行变异操作
            for j in range(NumTask):
                if locationnew[i, j] >bound[0][j] or locationnew[i, j] <bound[1][j]:
                    locationnew[i, j] = round(random.random()*(bound[0][j]-bound[1][j]))+bound[1][j]

        ###
        ## 改变部分5：交叉--根据交叉概率，决定变异结果是否保存
        ##
        # 交叉
        for i in range(pop):
            buf1 = np.random.rand(1, NumTask)
            buf2 = np.argsort(buf1)  # 获取正向排序后的下标
            for j in range(NumTask):
                if random.random() > FM and (j != buf2[0, j]):
                    locationnew1[i, j] = location[i, j].copy()
                else:
                    locationnew1[i, j] = locationnew[i, j].copy()

        ###
        ##改变部分6：对新方案进行处理，获取新的最优解
        ###
        # 选择
        [fall1, rdetail1] = fitness_func(pop, NumTask, NumMac, CostMoney, CostSource, Source, locationnew1)
        for i in range(pop):
            if fall1[i]<fall[i]:
                location[i] = locationnew1[i].copy()
                fall[i]=fall1[i].copy()



        f_best = min(fall)
        buf1 = np.where(fall == f_best)[0]
        buf1 = buf1[0]
        solution = copy.deepcopy(location[buf1, :])  # 当前最佳方案
        bestsofar.append(f_best)

    show_result(MaxIter, bestsofar)
    solution1 = np.random.randint(0, NumMac, size=[1, NumTask])
    solution1[0] = copy.deepcopy(solution)
    solution = copy.deepcopy(solution1)
    reala = fitness_func(1, NumTask, NumMac, CostMoney, CostSource, Source, solution)
    print_result(reala, solution)



if __name__ == '__main__':
    root_path = r'E:\OfficialWorkRemote\OfficialWork\SIOA\Drosophila optimization\SourceData\data_file\\'
    data_loader = DataLoader(root_path=root_path)
    data_dic = data_loader.data_generator(0)

    main(NumTask=data_dic['task'], NumMac=data_dic['mac'],
         CostMoney=np.array(data_dic['cost']).reshape(data_dic['mac'], data_dic['task']),
         CostSource=np.array(data_dic['source_cost']).reshape(data_dic['mac'], data_dic['task']),
         Source=np.array(data_dic['source']).reshape(1, data_dic['mac']))


