from src.envSim.cpuGpu import CpuGpu
import numpy as np
from src.envSim.timeSim import TimeHolder
from src.envSim.task import Task
from src.envSim.simParam import ParamHolder

class Node:
    def __init__(self,id,cpu_num,gpu_num):
        self.__id = id
        self.__cpu_num = cpu_num
        self.__gpu_num = gpu_num
        self.__cpu = CpuGpu(self.__cpu_num)
        self.__max_cpu = self.__cpu.get_info()
        self.__gpu = []
        self.__max_gpu = self.__gpu_num * self.__max_cpu / self.__cpu_num
        self.__online_task_list = []
        self.__offline_task_list = []
        for _ in range(self.__gpu_num):
            self.__gpu.append(CpuGpu(1))
        self.__success_offline_task_list = []

    def __add_success_offline_task_list(self, task: Task):
        self.__success_offline_task_list.append(task)

    def __get_success_offline_task_info(self):
        cpu = 0
        gpu = 0
        length = len(self.__success_offline_task_list)
        for task in self.__success_offline_task_list:
            cpu += task.get_cpu_info().sum()
            gpu += np.array(task.get_gpu_info()).sum()
        return length, cpu, gpu

    def get_id(self):
        return self.__id

    def set_id(self, id):
        self.__id = id

    def reset_node(self):
        self.__cpu = CpuGpu(self.__cpu_num)
        self.__gpu = []
        for _ in range(self.__gpu_num):
            self.__gpu.append(CpuGpu(1))

    def get_max_cpu(self):
        return self.__max_cpu

    def get_max_gpu(self):
        return self.__max_gpu

    def get_cpu_info(self,len = -1):
        return self.__cpu.get_info(len)

    def get_gpu_info(self,len = -1):
        temp = []
        for gpu in self.__gpu:
            temp.append(gpu.get_info(len))
        temp = np.array(temp)
        if len > 0:
            assert temp.shape == (self.__gpu_num, min(len,TimeHolder().get_time_left())), "gpu形状不符"
        return temp

    def get_online_task(self):
        return self.__online_task_list

    def get_offline_task(self):
        return self.__offline_task_list

    def set_task(self,task:Task,gpu_site ={}):
        temp_cpu = task.get_cpu_info()
        self.__cpu.set_info(temp_cpu)
        task_gpu = task.get_gpu_info()
        for key in gpu_site:
            self.__gpu[key].set_info(task_gpu[gpu_site[key]])
        task.set_task(self.__id,len(temp_cpu),gpu_site)
        if task.get_arrive_time() < 0:
            self.__online_task_list.append(task)
        else:
            self.__offline_task_list.append(task)

    def pop_task(self,task:Task):
        start = task.get_arrive_time()
        if start >= 0:
            start = task.get_start_time()
        self.__cpu.set_info(-task.get_cpu_info(),start)
        gpu_site = task.get_gpu_site()
        task_gpu = task.get_gpu_info()
        for key in gpu_site:
            self.__gpu[key].set_info(-task_gpu[gpu_site[key]],start)
        task.pop_task()


    def __check_resource(self):
        cpu = self.__cpu.check()
        gpu = []
        for temp in self.__gpu:
            gpu.append(temp.check())
        gpu = np.array(gpu)
        if cpu < -ParamHolder().zero or gpu.min() < -ParamHolder().zero:
            return False
        else:
            return True

    def check(self):
        if self.__check_resource():
            return []
        pop_list = []
        while self.__offline_task_list:
            task = self.__offline_task_list.pop()
            if task.get_task_len() <= (TimeHolder().get_time() - task.get_start_time()):
                self.__add_success_offline_task_list(task)
                continue
            self.pop_task(task)
            pop_list.append(task)
            if self.__check_resource():
                return pop_list
        while self.__online_task_list:
            task = self.__online_task_list.pop()
            self.pop_task(task)
            pop_list.append(task)
            if self.__check_resource():
                return pop_list
        return pop_list

    def get_success_num(self):
        while self.__offline_task_list:
            task = self.__offline_task_list.pop()
            self.__add_success_offline_task_list(task)
        return self.__get_success_offline_task_info()


