import math
from src.envSim.task import Task
from src.envSim.node import Node
from src.simParam import __time_can_predict__,__time_block_size__,__time_accurately_predict__,__cpu_gpu_rate__
import numpy as np
from abc import abstractmethod
from src.envSim.timeSim import TimeHolder
class Mem:
    def __init__(self):
        self.time = TimeHolder().get_time()
        self.mem = None

class Scheduler:
    def __init__(self,cluster:list[Node],can_predict,time_can_predict = __time_can_predict__,time_block_size = __time_block_size__):
        self.cluster = cluster
        self.__can_predict = can_predict
        self.__time_can_predict = time_can_predict #可预测时间长度，如=720，720 *10 = 7200s
        self.__time_block_size = time_block_size #单个预测值代表的时间长度，如=90，90*10 = 900s
        self.reschedule_num = 0
        self.success_num = 0
        self.fail_num = 0
        self.__task_len = 0
        self.__task_mem = {}
        self.__node_mem = {}
        self.node_cache_num = 0
        self.task_cache_num = 0
        self.node_no_cache_num = 0
        self.task_no_cache_num = 0
        self.rate = __cpu_gpu_rate__
    @abstractmethod
    def run(self,task:Task):
        pass

    def get_can_predict(self):
        return self.__can_predict

    def __deal_data(self,temp_cpu,temp_gpu,func):
        cpu = temp_cpu[0:__time_accurately_predict__]
        gpu = []
        for i in range(len(temp_gpu)):
            gpu.append(temp_gpu[i][0:__time_accurately_predict__])
        if __time_accurately_predict__ >= len(temp_cpu):
            return cpu,gpu
        temp_cpu = temp_cpu[__time_accurately_predict__:]
        split_indices = np.arange(self.__time_block_size, len(temp_cpu), self.__time_block_size)
        sub_arrays = np.split(temp_cpu, split_indices)
        cpu = np.concatenate((cpu, np.array([func(sub_array) for sub_array in sub_arrays])), axis=0)
        for i in range(len(temp_gpu)):
            temp_gpu_i = temp_gpu[i][__time_accurately_predict__:]
            split_indices = np.arange(self.__time_block_size, len(temp_gpu_i), self.__time_block_size)
            sub_arrays = np.split(temp_gpu_i, split_indices)
            gpu[i] = np.concatenate((gpu[i], np.array([func(sub_array) for sub_array in sub_arrays])), axis=0)
        gpu = np.array(gpu)
        return cpu,gpu

    def __return_task_mem(self,mem):
        cpu = mem[0][:self.__task_len]
        gpu = []
        for i in range(len(mem[1])):
            gpu.append(mem[1][i][:self.__task_len])
        return cpu,gpu

    def get_task_info(self,task:Task):
        task_mem = None
        temp_cpu = task.get_cpu_info(self.__can_predict)
        min_len = min(len(temp_cpu),self.__time_can_predict,TimeHolder().get_time_left())
        if min_len <= __time_accurately_predict__:
            self.__task_len = min_len
        else:
            self.__task_len = __time_accurately_predict__ + math.ceil((min_len - __time_accurately_predict__)/__time_block_size__)
        if task.get_arrive_time() >= 0: #为离线任务，每次获取的信息应该相同，可以写入缓存中加速
            if task.get_id() not in self.__task_mem:
                self.__task_mem[task.get_id()] = Mem()
            task_mem = self.__task_mem[task.get_id()]
            if task_mem.mem is not None:
                self.task_cache_num += 1
                return self.__return_task_mem(task_mem.mem)
        if self.__can_predict:
            temp_cpu = temp_cpu[:min_len]
            temp_gpu = task.get_gpu_info(self.__can_predict)
            temp = []
            for i in range(len(temp_gpu)):
                temp.append(temp_gpu[i][:min_len])
            cpu,gpu = self.__deal_data(temp_cpu,temp,np.max)
        else:
            cpu = task.get_cpu_info(self.__can_predict)
            gpu = task.get_gpu_info(self.__can_predict)
        if task_mem is None:
            return cpu,gpu
        task_mem.mem = (cpu,gpu)
        self.task_no_cache_num += 1
        return self.__return_task_mem(task_mem.mem)

    def get_node_info(self,node:Node):
        if node.get_id() not in self.__node_mem:
            self.__node_mem[node.get_id()] = Mem()
        node_mem = self.__node_mem[node.get_id()]
        if node_mem.time == TimeHolder().get_time() and node_mem.mem is not None:
            self.node_cache_num += 1
            return self.__return_task_mem(node_mem.mem)
        if self.__can_predict:
            temp_cpu = node.get_cpu_info(self.__time_can_predict)
            temp_gpu = node.get_gpu_info(self.__time_can_predict)
            cpu,gpu = self.__deal_data(temp_cpu,temp_gpu,np.min)
        else:
            cpu = node.get_cpu_info(1)
            gpu = node.get_gpu_info(1)
        node_mem.time = TimeHolder().get_time()
        node_mem.mem = (cpu, gpu)
        self.node_no_cache_num += 1
        return self.__return_task_mem(node_mem.mem)

    def set_task(self,node:Node,task:Task,gpu_site={}):
        node.set_task(task,gpu_site)
        self.__node_mem[node.get_id()].mem = None