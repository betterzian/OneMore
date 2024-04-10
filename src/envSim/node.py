from src.envSim.cpuGpu import Cpu, Gpu
from src.envSim.timeSim import TimeHolder
from src.envSim.task import Task


class Node:
    def __init__(self, id, cpu_num, gpu_num):
        self.__id = id
        self.__cpu_num = cpu_num
        self.__gpu_num = gpu_num
        self.__cpu = Cpu(self.__cpu_num)
        self.__max_cpu = self.__cpu.get_info()
        self.__gpu = Gpu(self.__gpu_num)
        self.__max_gpu = self.__gpu.get_info()
        self.__online_task_list = []
        self.__offline_task_list = []
        self.__success_offline_task_list = []

    def __add_success_offline_task_list(self, task: Task):
        self.__success_offline_task_list.append(task)

    def __get_success_offline_task_info(self):
        cpu = 0
        gpu = 0
        length = len(self.__success_offline_task_list)
        for task in self.__success_offline_task_list:
            cpu += task.get_cpu_info().sum()
            gpu += task.get_gpu_info().sum()
        return length, cpu, gpu

    def get_id(self):
        return self.__id

    def set_id(self, id):
        self.__id = id

    def reset_node(self):
        self.__cpu = Cpu(self.__cpu_num)
        self.__max_cpu = self.__cpu.get_info()
        self.__gpu = Gpu(self.__gpu_num)
        self.__max_gpu = self.__gpu.get_info()
        self.__online_task_list = []
        self.__offline_task_list = []
        self.__success_offline_task_list = []

    def get_max_cpu(self):
        return self.__max_cpu

    def get_max_gpu(self):
        return self.__max_gpu

    def get_cpu_info(self, length=-1):
        return self.__cpu.get_info(length).reshape(-1)

    def get_gpu_info(self, length=-1):
        return self.__gpu.get_info(length)

    def get_online_task(self):
        return self.__online_task_list

    def get_offline_task(self):
        return self.__offline_task_list

    def set_task(self, task: Task, gpu_site=None):
        if gpu_site is None:
            gpu_site = {}
        task_cpu = task.get_cpu_info()
        self.__cpu.set_info(task.get_cpu_info())
        self.__gpu.set_info(task.get_gpu_info(), dic=gpu_site)
        task.set_task(self.__id, len(task_cpu), gpu_site)
        if task.get_arrive_time() < 0:
            self.__online_task_list.append(task)
        else:
            self.__offline_task_list.append(task)

    def pop_task(self, task: Task):
        start = task.get_arrive_time()
        if start >= 0:
            start = task.get_start_time()
        self.__cpu.set_info(-task.get_cpu_info(), start)
        self.__gpu.set_info(-task.get_gpu_info(), start, task.get_gpu_site())
        task.pop_task()

    def __check_resource(self, all_bool):
        cpu = self.__cpu.check(all_bool)
        gpu = self.__gpu.check(all_bool)
        return not (cpu or gpu)

    def check(self, all_bool=False):
        if self.__check_resource(all_bool):
            return []
        pop_list = []
        while self.__offline_task_list:
            task = self.__offline_task_list.pop()
            if task.get_task_len() <= (TimeHolder().get_time() - task.get_start_time()):
                self.__add_success_offline_task_list(task)
                continue
            self.pop_task(task)
            pop_list.append(task)
            if self.__check_resource(all_bool):
                return pop_list
        while self.__online_task_list:
            task = self.__online_task_list.pop()
            self.pop_task(task)
            pop_list.append(task)
            if self.__check_resource(all_bool):
                return pop_list
        return pop_list

    def get_success_num(self):
        if TimeHolder().get_time() == 0:
            return 0, 0, 0
        while self.__offline_task_list:
            task = self.__offline_task_list.pop()
            self.__add_success_offline_task_list(task)
        return self.__get_success_offline_task_info()
