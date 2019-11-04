import matplotlib.pyplot as plt
import numpy as np
import copy



def show_result(MaxIter, bestsofar):
    plt.plot(range(MaxIter), bestsofar, '-y')
    plt.title('The best of process in foa')
    plt.xlabel('iter')
    plt.ylabel('fitness')
    plt.show()

def print_result(reala, solution):
    print('the result is showing...')
    print('best_fitness for min problem:', reala[0][0])
    print('best solution for result:', solution)
    print('best solution detail for result:', reala[1])

def fitness_func(NumTask, NumMac, CostMoney, CostSource, Source, gene):  # 计算目标值

    ###
    ##部分3：fitness_func()几个存放数据的数组初始化
    ###
    buf = np.shape(gene)  # 多少行数据 ndarray(计算初代localtion的形状，第一次循环为100*20)
    fall = []
    detailall = []

    for i in range(buf[0]):  # 对每一个方案进行计算(buf[0]的值，第一次循环为100，以后全为1，buf[1]全为20)
        ###
        ##部分4：方案可行性检验和信息存储
        ###
        f = 0   # 放置成本
        detail = np.random.randint(-1, 0, size=[3, NumTask])
        buf1 = [CostMoney[gene[i, j], j] for j in range(buf[1])]
        buf2 = np.argsort(buf1) # 获取正向排序后的下标；argsort函数返回的是数组值从小到大的索引值
        buf3 = copy.deepcopy(Source[0][:])  # 缓存内存(深复制独立内存，机器拥有资源数量)
        for j in range(buf[1]): # 按照成本耗费来决定是否进行（进行20次循环）
            if buf3[gene[i, buf2[j]]]-CostSource[gene[i, buf2[j]], buf2[j]] >= 0:
                detail[0, buf2[j]] = gene[i, buf2[j]]  #detail第一行，存放每个任务交给谁来做
                detail[1, buf2[j]] = CostSource[gene[i, buf2[j]], buf2[j]]  #detail第二行，存放相应消耗资源
                detail[2, buf2[j]] = CostMoney[gene[i, buf2[j]], buf2[j]]  #detail第三行，存放消耗的钱财
                buf3[gene[i, buf2[j]]] = buf3[gene[i, buf2[j]]] - CostSource[gene[i, buf2[j]], buf2[j]]  #更新一下，每个机器还剩多少资源

        ###
        ##部分5：未分配机器的任务的二次分配
        ###
        buf4 = list(detail[0])
        buf4 = list([idx for (idx, val) in enumerate(buf4) if val == -1])  # 查看下还没有被服务的点
        #enumerate()函数，在迭代一个序列的同时跟踪正在被处理的元素索引。传入参数idx 和 val ；其中idx为序列索引；val为返回值
        if len(buf4) >= 1 :
            for j in buf4: # 对每一个还未满足的进行二次选择
                #  找出可以满足的对象
                buf5 = list([idx for (idx, val) in enumerate(buf3) if val >= CostSource[gene[i, j], j]])  # 选出资源还充足的机器（有点漏洞）
                buf6 = list([CostMoney[idx, j] for (idx, val) in enumerate(buf3) if val >= CostSource[gene[i, j], j]])  # 选出资源充足机器完成任务要花的钱
                if len(buf5) >= 1:#如果有符合要求，还剩资源的机器则进入
                    buf7 = np.where(buf6 == min(buf6))[0]  #选出消耗钱最少的那个机器
                    detail[0, j] = buf5[buf7[0]]   #把缺失任务交给谁做分配出去，下面是相应的更新
                    detail[1, j] = CostSource[detail[0, j], j]
                    detail[2, j] = CostMoney[detail[0, j], j]
                    buf3[detail[0, j]] = buf3[detail[0, j]] - CostSource[detail[0, j], j]
                else :#有任务没有机器能解决，输出无解
                    print('无解')
                    f = 100000000
                    return fall, detailall

        ###
        ##部分6：统计成本信息和方案数据
        ###
        f = sum(detail[2, :])  #统计一共消耗了多少钱
        fall.append(f)  #把本次情况消耗的钱丢进去  共100种情况
        detailall.append(detail)  #把本次情况的机器分配情况丢进去  共100种情况

    ###
    ##部分7：回到调用点
    ###
    return fall, detailall

def main():
    ###
    ##部分1：算法数据的初始化
    ###
    # 随机生成数据
    NumTask : int = 20  # 需要处理任务的数量
    NumMac: int = 10  # 可以处理任务的机器数量
    CostMoney = np.random.randint(1, 11, size=[NumMac, NumTask])  # 每个机器对应每个任务所耗费的成本（追求成本最低）
    CostSource = np.random.randint(10, 100, size=[NumMac, NumTask])  # 每个机器对应每个任务所耗费的资源（追求资源不超过）
    Source = np.random.randint(100, 1000, size=[1, NumMac])  # 每个机器所拥有的资源
    # 算法参数
    MaxIter: int = 1000  #最大迭代次数
    pop : int = 5 #
    # 变量的取值范围
    bound = np.random.randint(1, 2, size=[2, NumTask]) #全1的2*20数组
    bound[0] = bound[0] * (NumMac - 1)  # 上边界，每个变量的上边界可以不同   上边界全9
    bound[1] = bound[1] * 0  # 下边界，每个变量的下边界可以不同        下边界全0
    # 生成初始代
    location = np.random.randint(0, NumMac, size=[100, NumTask])

    ###
    ##部分2：跳转fitness_func（）的语句
    ###
    [buf, rdetail] = fitness_func(NumTask, NumMac, CostMoney, CostSource, Source, location)

    ###
    ##部分8：记录初始100种方案的最优方案
    ###
    f_best = min(buf)  #100种情况里，费钱最少的一共是花了多少
    buf1 = np.where(buf == f_best)[0]
    buf1 = buf1[0]  #成本最少的那个的下标
    solution = copy.deepcopy(location[buf1, :]) # 当前最佳方案的初始情况
    detail = copy.deepcopy((rdetail[buf1]))  #当前最佳方案的完善情况（处理完分配超量的情况，且附带其他完整信息）

    ###
    ##部分9：进入重新迭代选取更优方案
    ###
    bestsofar = [] # 最佳值的变化
    for itera in range(MaxIter):  # 运行 代数
        print(itera)
        for i in range(pop):#pop个果蝇向最优解果蝇集中后，再把他们随机出去，要随机5次
            ###
            ##部分10：随机改变方案并限制不能越界
            ###
            newss = np.sum([solution, np.random.randint(-2, 3, size=[1, NumTask])], axis = 0)
            for j in range(NumTask):
                if newss[0, j]> bound[0, j]:
                    newss[0, j] = bound[0, j]
                elif newss[0, j]< bound[1, j]:
                    newss[0, j] = bound[1, j]

            ###
            ##部分11：对新方案进行处理，返回成本和方案信息，并更新最优解
            ###
            reala = fitness_func(NumTask, NumMac, CostMoney, CostSource, Source, newss)  #重新挪动位置后，进行新一轮的超量情况处理
            if reala[0]<f_best:
                f_best = reala[0][0]
                solution = newss[:]
        ###
        ##部分12：记录最优解的成本变化历程
        ###
        bestsofar.append(f_best) #更新新的最低成本

    ###
    ##部分13：打印结果
    ###
    show_result(MaxIter, bestsofar)
    solution1 = np.random.randint(0, NumMac, size=[1, NumTask])
    solution1[0] = copy.deepcopy(solution)
    solution = copy.deepcopy(solution1)

    reala = fitness_func(NumTask, NumMac, CostMoney, CostSource, Source, solution)
    print_result(reala, solution)



if __name__ == '__main__':
    main()
