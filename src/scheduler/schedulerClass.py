import math
from src.envSim.task import Task
from src.envSim.node import Node
from src.envSim.simParam import ParamHolder
import numpy as np
from abc import abstractmethod
from src.envSim.timeSim import TimeHolder
import time


class Mem:
    def __init__(self):
        self.time = TimeHolder().get_time()
        self.mem = None


class Scheduler:
    def __init__(self, cluster: list[Node], can_predict, task_mem, node_mem,
                 time_can_predict=None, time_block_size=None):
        if time_can_predict is None:
            time_can_predict = ParamHolder().time_can_predict
        if time_block_size is None:
            time_block_size = ParamHolder().time_block_size
        self.cluster = cluster
        self.__can_predict = can_predict
        self.__time_can_predict = time_can_predict  # 可预测时间长度，如=720，720 *10 = 7200s
        self.__time_block_size = time_block_size  # 单个预测值代表的时间长度，如=90，90*10 = 900s
        self._reschedule_num = 0
        self._success_num = 0
        self._fail_num = 0
        self._task_len = 0
        self._task_mem = task_mem
        self._node_mem = node_mem
        self._node_cache_num = 0
        self._task_cache_num = 0
        self._node_no_cache_num = 0
        self._task_no_cache_num = 0
        self._rate = ParamHolder().cpu_gpu_rate
        self._time = 0
        self._force_num = 0

    @abstractmethod
    def run(self, task: Task):
        pass

    def add_reschedule_num(self, num=1):
        self._reschedule_num += num

    def add_fail_num(self, num=1):
        self._fail_num += num

    def get_can_predict(self):
        return self.__can_predict

    def set_time(self):
        self._time = time.time() - self._time

    def get_time(self):
        return self._time

    def get_reschedule_num(self):
        return self._reschedule_num

    def get_fail_num(self):
        return self._fail_num

    def get_task_len(self):
        return self._task_len

    def get_node_cache_num(self):
        return self._node_cache_num

    def get_task_cache_num(self):
        return self._task_cache_num

    def get_node_no_cache_num(self):
        return self._node_no_cache_num

    def get_task_no_cache_num(self):
        return self._task_no_cache_num

    def get_force_num(self):
        return self._force_num

    def __deal_data(self, temp_cpu, temp_gpu, func):
        cpu = temp_cpu[0:ParamHolder().time_accurately_predict]
        gpu = temp_gpu[:,0:ParamHolder().time_accurately_predict]
        if ParamHolder().time_accurately_predict >= len(temp_cpu):
            return cpu, gpu
        temp_cpu = temp_cpu[ParamHolder().time_accurately_predict:]
        length = len(temp_cpu)
        split_indices = np.arange(self.__time_block_size, length, self.__time_block_size)
        sub_arrays = np.split(temp_cpu, split_indices)
        cpu = np.concatenate((cpu, np.array([func(sub_array) for sub_array in sub_arrays])), axis=0)
        if temp_gpu.size > 0:
            temp_gpu = temp_gpu[:,ParamHolder().time_accurately_predict:]
            split_indices = np.arange(self.__time_block_size, length, self.__time_block_size)
            sub_arrays = np.split(temp_gpu, split_indices,axis=1)
            gpu = np.concatenate((gpu, np.array([func(sub_array,axis=1) for sub_array in sub_arrays]).transpose()), axis=1)
        return cpu, gpu

    def __return_task_mem(self, mem):
        cpu = mem[0][:self._task_len]
        gpu = mem[1]
        if gpu.size > 0:
            gpu = gpu[:,:self._task_len]
        return cpu.copy(), gpu.copy()

    def get_task_info(self, task: Task):
        task_mem = None
        temp_cpu = task.get_cpu_info(self.__can_predict)
        min_len = min(len(temp_cpu), self.__time_can_predict, TimeHolder().get_time_left())
        if min_len <= ParamHolder().time_accurately_predict:
            self._task_len = min_len
        else:
            self._task_len = ParamHolder().time_accurately_predict + math.ceil(
                (min_len - ParamHolder().time_accurately_predict) / ParamHolder().time_block_size)
        if task.get_arrive_time() >= 0:  # 为离线任务，每次获取的信息应该相同，可以写入缓存中加速
            if task.get_id() not in self._task_mem:
                self._task_mem[task.get_id()] = Mem()
            task_mem = self._task_mem[task.get_id()]
            if task_mem.mem is not None:
                self._task_cache_num += 1
                return self.__return_task_mem(task_mem.mem)
        if self.__can_predict:
            temp_cpu = temp_cpu[:min_len]
            temp_gpu = task.get_gpu_info(self.__can_predict)[:,:min_len]
            cpu, gpu = self.__deal_data(temp_cpu, temp_gpu, np.max)
        else:
            cpu = task.get_cpu_info(self.__can_predict)
            gpu = task.get_gpu_info(self.__can_predict)
        if gpu.size == 0:
            gpu = np.array([])
        if task_mem is None:
            return cpu.copy(), gpu.copy()
        task_mem.mem = (cpu, gpu)
        self._task_no_cache_num += 1
        return self.__return_task_mem(task_mem.mem)

    def get_node_info(self, node: Node):
        if node.get_id() not in self._node_mem:
            self._node_mem[node.get_id()] = Mem()
        node_mem = self._node_mem[node.get_id()]
        if node_mem.time == TimeHolder().get_time() and node_mem.mem is not None:
            self._node_cache_num += 1
            return self.__return_task_mem(node_mem.mem)
        if self.__can_predict:
            temp_cpu = node.get_cpu_info(self.__time_can_predict)
            temp_gpu = node.get_gpu_info(self.__time_can_predict)
            cpu, gpu = self.__deal_data(temp_cpu, temp_gpu, np.min)
        else:
            cpu = node.get_cpu_info(1)
            gpu = node.get_gpu_info(1)
        if gpu.size == 0:
            gpu = np.array([])
        node_mem.time = TimeHolder().get_time()
        node_mem.mem = (cpu, gpu)
        self._node_no_cache_num += 1
        return self.__return_task_mem(node_mem.mem)

    def set_task(self, node: Node, task: Task, gpu_site={}):
        node.set_task(task, gpu_site)
        self._node_mem[node.get_id()].mem = None

    def force_set_online_task(self, task: Task):
        priority = 999999999
        task_cpu = task.get_cpu_info(self.__can_predict)
        task_cpu = task_cpu[:ParamHolder().time_accurately_predict]
        now_select = -1
        for node in self.cluster:
            temp_priority = 0
            temp_node_cpu = node.get_cpu_info(ParamHolder().time_accurately_predict)
            if not self.__can_predict:
                temp_node_cpu = temp_node_cpu[:1]
            offline_task = node.get_offline_task()
            for temp_task in reversed(offline_task):
                waste_time = (TimeHolder().get_time() - temp_task.get_start_time())
                temp_priority += temp_task.get_task_weight() * waste_time
                temp_cpu = temp_task.get_cpu_info()[waste_time:]
                temp_cpu = temp_cpu[:min(ParamHolder().time_accurately_predict, TimeHolder().get_time_left())]
                if not self.__can_predict:
                    temp_cpu = temp_cpu[:1]
                temp_node_cpu = np.concatenate(
                    (temp_node_cpu[:len(temp_cpu)] + temp_cpu, temp_node_cpu[len(temp_cpu):]), axis=0)
                if np.any(task_cpu > temp_node_cpu):
                    continue
                else:
                    break
            if np.any(task_cpu > temp_node_cpu):
                continue
            if temp_priority < priority:
                priority = temp_priority
                now_select = node
        assert now_select != -1, "在线任务无法放置"
        self.set_task(now_select, task, {})
        self._force_num += 1
        return now_select.check()

    def get_all_node_info(self):
        node_info = np.zeros((len(self.cluster), 9))
        for i in range(len(self.cluster)):
            cpu, gpu = self.get_node_info(self.cluster[i])
            node_info[i][0] = cpu[0]
            if len(gpu) != 0:
                gpu = gpu[:, 0].reshape(-1)
                node_info[i][1:1+len(gpu)] = gpu
        return node_info
