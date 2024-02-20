import csv
import numpy as np
from random import randint
import pandas as pd
from src.envSim.simParam import ParamHolder
from src.envSim.generateEnv import generate_offline_task_list

class Robot:
    def __init__(self, epoch, fail_num, task):
        self.__epoch = epoch
        self.__fail_num = fail_num
        self.__task = task

    def run(self, task=None):
        for _ in range(self.__epoch):
            fail_num = 0
            node_cpu = np.array(1)
            node_gpu = np.array(1)
            while fail_num < self.__fail_num:
                task = self.__task[randint(0, len(self.__task))]
                task_cpu, task_gpu = self.get_task_info(task)
                if node_cpu >= task_cpu and node_gpu >= task_gpu:
                    node_cpu -= task_cpu
                    node_gpu -= task_gpu
                else:
                    fail_num += 1
            data = np.concatenate((node_cpu, node_gpu, node_cpu.sum() + node_gpu.sum() * ParamHolder().cpu_gpu_rate),axis=1)
            data = pd.DataFrame(data)
            data.to_csv('../data_src/offline_task/off_task_list_data' + '.txt', index=False, header=True, sep='\t',
                        mode='a', encoding="utf-8", quoting=csv.QUOTE_NONE, escapechar=',')


def start(robot):
    robot.run()


def gather_data(robot_num=50, epoch=10000, test=False):
    task = generate_offline_task_list(all=True)
    fail_num = len(task) * 0.05
    robot_list = [Robot(epoch, fail_num, task) for _ in range(robot_num)]
    if test:
        for i in range(robot_num):
            start(robot_list[i])
        return
    from multiprocessing import Pool
    p = Pool(robot_num)
    for i in range(robot_num):
        p.apply_async(start(robot_list[i]), args=())
    p.close()
    p.join()


if __name__ == "__main__":
    gather_data(test=True)
