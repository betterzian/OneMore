from src.envSim.task import Task
from src.envSim.node import Node
from src.simParam import __time_can_predict__,__time_block_size__,__time_accurately_predict__
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
    @abstractmethod
    def run(self,task:Task):
        pass

    def get_task_info(self,task:Task):
        if task.get_id() not in self.__task_mem:
            self.__task_mem[task.get_id()] = Mem()
        task_mem = self.__task_mem[task.get_id()]
        if task_mem.time == TimeHolder().get_time() and task_mem.mem != None:
            return task_mem.mem
        if self.__can_predict:
            temp_cpu = task.get_cpu_info(self.__can_predict)
            self.__task_len = len(temp_cpu)
            temp_cpu = temp_cpu[:min(len(temp_cpu)+1,self.__time_can_predict)]
            temp_gpu = task.get_gpu_info(self.__can_predict)
            temp_gpu = temp_gpu[:min(len(temp_cpu)+1,self.__time_can_predict)]
            split_indices = np.arange(self.__time_block_size, len(temp_cpu), self.__time_block_size)
            sub_arrays = np.split(temp_cpu, split_indices)
            temp_cpu = temp_cpu[0:__time_accurately_predict__]
            cpu = np.array([np.max(sub_array) for sub_array in sub_arrays])
            cpu = np.concatenate((temp_cpu,cpu),axis=0)
            gpu = []
            for i in range(len(temp_gpu)):
                split_indices = np.arange(self.__time_block_size, len(temp_gpu[i]), self.__time_block_size)
                sub_arrays = np.split(temp_gpu[i], split_indices)
                temp_gpu_i = temp_gpu[i][0:__time_accurately_predict__]
                gpu.append(np.concatenate((temp_gpu, np.array([np.max(sub_array) for sub_array in sub_arrays])), axis=0))
            gpu = np.array(gpu)
        else:
            cpu = task.get_cpu_info(self.__can_predict)
            gpu = task.get_gpu_info(self.__can_predict)
        task_mem.time = TimeHolder().get_time()
        task_mem.mem = (cpu,gpu)
        return task_mem.mem

    def get_node_info(self,node:Node):
        if node.get_id() not in self.__node_mem:
            self.__node_mem[node.get_id()] = Mem()
        node_mem = self.__node_mem[node.get_id()]
        if node_mem.time == TimeHolder().get_time() and node_mem.mem != None:
            return node_mem.mem
        if self.__can_predict:
            length = min(self.__time_can_predict,self.__task_len)
            temp_cpu = node.get_cpu_info(self.__time_can_predict)
            temp_cpu = temp_cpu[:min(len(temp_cpu) + 1, self.__time_can_predict)]
            temp_gpu = node.get_gpu_info(self.__time_can_predict)
            split_indices = np.arange(self.__time_block_size, len(temp_cpu), self.__time_block_size)
            sub_arrays = np.split(temp_cpu, split_indices)
            temp_cpu = temp_cpu[0:__time_accurately_predict__]
            cpu = np.concatenate((temp_cpu,np.array([np.mean(sub_array) for sub_array in sub_arrays])),axis=0)
            gpu = []
            for i in range(len(temp_gpu)):
                split_indices = np.arange(self.__time_block_size, len(temp_gpu[i]), self.__time_block_size)
                sub_arrays = np.split(temp_gpu[i], split_indices)
                temp_gpu_i = temp_gpu[i][0:__time_accurately_predict__]
                gpu.append(np.concatenate((temp_gpu_i, np.array([np.mean(sub_array) for sub_array in sub_arrays])), axis=0))
            gpu = np.array(gpu)
        else:
            cpu = node.get_cpu_info(1)
            gpu = node.get_gpu_info(1)
        node_mem.time = TimeHolder().get_time()
        node_mem.mem = (cpu, gpu)
        return node_mem.mem

    def set_task(self,node:Node,task:Task,gpu_site={}):
        node.set_task(task,gpu_site)
        self.__node_mem[node.get_id()].mem = None