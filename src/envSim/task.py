from src.envSim.cpuGpu import CpuGpu
from src.envSim.timeSim import TimeHolder
import numpy as np
import math
class Task:
    def __init__(self, id, cpu, gpu= 0, time_len = -1, arrive_time = -1):
        self.__id = id
        if time_len < 0: #为在线任务
            self.__max_cpu = cpu[0]
            cpu = cpu[1:]
            time_flag = TimeHolder().get_time_init_flag()
            self.__time_len = len(cpu)
        else: #为离线任务
            self.__max_cpu = cpu
            time_flag = 0
            self.__time_len = time_len
        self.__cpu = CpuGpu(cpu, self.__time_len, time_flag)
        self.__max_gpu = gpu
        self.__gpu = []
        self.__start_time = 0
        self.__arrive_time = arrive_time
        self.__gpu_site = []
        while(gpu > 0):
            self.__gpu.append(CpuGpu(1, self.__time_len, time_flag))
            gpu -= 1
        self.__gpu_num = len(self.__gpu)

    def get_id(self):
        return self.__id

    def set_start_time(self):
        self.__start_time = TimeHolder().get_time()

    def get_start_time(self):
        return self.__start_time

    def get_arrive_time(self):
        return self.__arrive_time

    def get_task_len(self):
        return self.__time_len

    def get_cpu_info(self,canPredict = True):
        if canPredict:
            temp_cpu = self.__cpu.get_info()
            if self.__arrive_time < 0:
                return temp_cpu[-TimeHolder().get_time_left():]
            return temp_cpu[0:min(len(temp_cpu),TimeHolder().get_time_left())]
        else:
            return np.array([self.__max_cpu])



    def get_gpu_info(self,canPredict = True):
        temp = []
        if canPredict:
            length = min(len(self.__cpu.get_info()), TimeHolder().get_time_left())
            for gpu in self.__gpu:
                temp_gpu = gpu.get_info()
                temp.append(temp_gpu[0:length])
            temp = np.array(temp)
            if len(self.__gpu) > 0:
                assert temp.shape == (self.__gpu_num,length), "task,gpu形状不符"
            return temp
        else:
            for gpu in self.__gpu:
                temp.append([gpu.get_info()[0]])
            temp = np.array(temp)
            if len(self.__gpu) > 0:
                assert temp.shape == (self.__gpu_num, 1), "task,gpu形状不符"
            return temp


    def set_gpu_site(self,gpu_site ={}):
        self.__gpu_site = gpu_site

    def get_gpu_site(self):
        return self.__gpu_site
