import os
import numpy as np


class DataLoader:
    def __init__(self,
                 root_path=r'E:\OfficialWorkRemote\OfficialWork\SIOA\Drosophila optimization\SourceData\data_file\\'):
        self.file_path_container = [root_path + file_path for file_path in os.listdir(root_path)]

        self.num_mac_container = []
        self.num_task_container = []
        self.cost_container = []
        self.source_cost_container = []
        self.source_container = []

        # 数据处理主程序
        self.file_process()

    def file_read(self, file_path):
        tmp_container = []
        num_macs = 0
        cost__flag = True
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                split_line = line.split(' ')
                if len(split_line) < 2:
                    continue
                if len(split_line) == 2:
                    self.num_mac_container.append(split_line[0])
                    self.num_task_container.append(split_line[1])
                    num_macs = int(split_line[0])
                    continue
                if len(split_line) != num_macs:
                    tmp_container.append(split_line)
                    if len(tmp_container) >= num_macs:
                        if cost__flag:
                            self.cost_container.append(tmp_container)
                            cost__flag = False
                            tmp_container = []
                            continue
                        self.source_cost_container.append(tmp_container)
                        cost__flag = True
                        tmp_container = []
                    continue
                if len(split_line) == num_macs:
                    self.source_container.append(split_line)

    def file_process(self):
        for file_path in self.file_path_container:
            self.file_read(file_path)
        self.format_process()

    def format_process(self):
        self.num_mac_container = np.array(self.num_mac_container).astype(int).tolist()
        self.num_task_container = np.array(self.num_task_container).astype(int).tolist()
        self.cost_container = [np.array(val).astype(int).tolist() for val in self.cost_container]
        self.source_cost_container = [np.array(val).astype(int).tolist() for val in self.source_cost_container]
        self.source_container = [np.array(val).astype(int).tolist() for val in self.source_container]

    def get_data_len(self):
        return len(self.num_task_container)

    # 输入要获取的数据index 返回数据（dic形式）
    def data_generator(self, index):
        res_dic = {'mac': self.num_mac_container[index],
                   'task': self.num_task_container[index],
                   'cost': self.cost_container[index],
                   'source_cost': self.source_cost_container[index],
                   'source': self.source_container[index]
                   }
        return res_dic


if __name__ == '__main__':
    root_path = r'E:\OfficialWorkRemote\OfficialWork\SIOA\Drosophila optimization\SourceData\data_file\\'
    data_loader = DataLoader(root_path=root_path)

    # 测试数据输出
    data_res = data_loader.data_generator(0)
    print(data_res)