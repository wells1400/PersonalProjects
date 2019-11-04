import matplotlib.pyplot as plt
import numpy as np
import copy
import math
import random



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

def fitness_func(pop, NumTask, NumMac, CostMoney, CostSource, Source, gene):  # 计算目标值
    ###
    ##改变部分7：多传入了一个pop值
    ###
    buf= []
    buf.append(pop)  # 种群大小
    buf.append(NumTask)  # 任务数量  buf=[50,5]
    fall = []
    detailall = []
    for i in range(buf[0]):  # 对每一个方案进行计算
        f = 0
        detail = np.random.randint(-1, 0, size=[3, NumTask])
        buf1 = [CostMoney[gene[i, j], j] for j in range(buf[1])]  #存放成本值
        buf2 = np.argsort(buf1) # 获取正向排序后的下标
        buf3 = copy.deepcopy(Source[0][:])  # 缓存内存
        for j in range(buf[1]): # 按照成本耗费来决定是否进行
            if buf3[gene[i, buf2[j]]]-CostSource[gene[i, buf2[j]], buf2[j]] >= 0:
                detail[0, buf2[j]] = gene[i, buf2[j]]
                detail[1, buf2[j]] = CostSource[gene[i, buf2[j]], buf2[j]]
                detail[2, buf2[j]] = CostMoney[gene[i, buf2[j]], buf2[j]]
                buf3[gene[i, buf2[j]]] = buf3[gene[i, buf2[j]]] - CostSource[gene[i, buf2[j]], buf2[j]]
        buf4 = list(detail[0])
        buf4 = list([idx for (idx, val) in enumerate(buf4) if val == -1])  # 查看下还没有被服务的点
        if len(buf4) >= 1 :
            for j in buf4: # 对每一个还未满足的进行二次选择
                #  找出可以满足的对象
                buf5 = list([idx for (idx, val) in enumerate(buf3) if val >= CostSource[gene[i, j], j]])  # 查看下还没有被服务的点
                buf6 = list([CostMoney[idx, j] for (idx, val) in enumerate(buf3) if val >= CostSource[gene[i, j], j]])  # 查看下还没有被服务的点
                if len(buf5) >= 1:
                    buf7 = np.where(buf6 == min(buf6))[0]
                    detail[0, j] = buf5[buf7[0]]
                    detail[1, j] = CostSource[detail[0, j], j]
                    detail[2, j] = CostMoney[detail[0, j], j]
                    buf3[detail[0, j]] = buf3[detail[0, j]] - CostSource[detail[0, j], j]
                else :
                    print('无解')
                    f = 100000000
                    return fall, detailall
        f = sum(detail[2, :])
        fall.append(f)
        detailall.append(detail)
    return fall, detailall

def main():
    ###
    ##改变部分1：初始化数据的改动和增加
    ###
    # 随机生成数据
    NumTask : int = 5  # 需要处理任务的数量
    NumMac: int = 5  # 可以处理任务的机器数量
    CostMoney = np.random.randint(1, 11, size=[NumMac, NumTask])  # 每个机器对应每个任务所耗费的成本
    CostSource = np.random.randint(10, 100, size=[NumMac, NumTask])  # 每个机器对应每个任务所耗费的资源
    Source = np.random.randint(100, 1000, size=[1, NumMac])  # 每个机器对应每个任务所耗费的资源
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


    f_best = min(fall)
    buf1 = np.where(fall == f_best)[0]
    buf1 = buf1[0]
    solution = copy.deepcopy(location[buf1, :]) # 当前最佳方案
    detail = copy.deepcopy((rdetail[buf1]))

    ###
    ##改变部分2：两个进行变异、交叉时需要使用的数组
    ###
    locationnew = copy.deepcopy(location)
    locationnew1 = copy.deepcopy(location)
    bestsofar = [] # 最佳值的变化

    ###
    ##改变部分3：迭代逻辑
    ###
    for itera in range(MaxIter):  # 运行 代数
        print(itera)
        rf = FC*math.pow(2, math.exp(1-MaxIter/(1+MaxIter-itera)))  #一不好玩的数学公式吧

        ###
        ##改变部分4：变异--随机选取任意一个方案，进行位置挪动
        ###
        # 变异
        for i in range(pop): # 对一个个体
            buf = [j for j in range(pop)]  #存放0到49的一个列表
            buf.remove(buf[i])  #把下标为i的删掉
            buf1 = np.random.rand(1, pop-1)  #存放49个随机数
            buf2 = np.argsort(buf1)  # 获取正向排序后的下标
            locationnew[i] = location[buf[buf2[0, 0]]] + rf*(location[buf[buf2[0, 1]]]-location[buf[buf2[0, 2]]])#所谓buf[buf2[0,0]]或[0,1]就是一个随机元素
            locationnew[i] = locationnew[i].astype(np.int32)  #对随机选取的行进行变异操作
            for j in range(NumTask):
                if locationnew[i, j] >bound[0][j] or locationnew[i, j] <bound[1][j]:
                    locationnew[i, j] = round(random.random()*(bound[0][j]-bound[1][j]))+bound[1][j]

        ###
        ##改变部分5：交叉--根据交叉概率，决定变异结果是否保存
        ###
        # 交叉
        for i in range(pop):
            buf1 = np.random.rand(1, NumTask)
            buf2 = np.argsort(buf1)  # 获取正向排序后的下标
            for j in range(NumTask):
                if random.random()>FM and (j != buf2[0,j]):
                    locationnew1[i,j]=location[i,j].copy()
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
    main()
