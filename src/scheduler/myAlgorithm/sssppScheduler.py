from src.scheduler.schedulerClass import Scheduler
from src.envSim.simParam import ParamHolder
import numpy as np
from copy import deepcopy

class SSSPPScheduler(Scheduler):
    def __init__(self,cluster,can_predict = True,task_mem = {},node_mem = {}):
        super().__init__(cluster,can_predict,task_mem,node_mem)
        self.state_int = np.loadtxt("../data_src/state_value/"+ParamHolder().filename+"/state_int.csv", delimiter=",")
        self.state_float = np.loadtxt("../data_src/state_value/"+ParamHolder().filename+"/state_float.csv", delimiter=",")
        self.state_only_float = np.loadtxt("../data_src/state_value/"+ParamHolder().filename+"/state_only_float.csv", delimiter=",")

    def run(self, task):
        task_cpu, task_gpu = self.get_task_info(task)

        task_gpu = task_gpu[:, 0]
        now_priority = -1e8
        now_select = -1
        gpu_select = -1
        for node in self.cluster:
            select = -1
            temp_select = {}
            temp_node_cpu, temp_node_gpu = self.get_node_info(node)
            temp_node_gpu = temp_node_gpu[:, 0]
            if np.any(task_cpu > temp_node_cpu):
                continue
            else:
                task_cpu = task_cpu.max()
                temp_node_cpu = temp_node_cpu.min()
                temp_node_gpu_old = deepcopy(temp_node_gpu)
                gpu_int_old = np.count_nonzero(temp_node_gpu_old == 1)
                gpu_float_old = np.count_nonzero(temp_node_gpu_old > ParamHolder().zero) - gpu_int_old
                if gpu_float_old > 0:
                    if gpu_int_old > 0:
                        state_old = self.state_int[int(task_cpu)][int(gpu_int_old)]
                        temp = -1e8
                        for j in range(len(temp_node_gpu)):
                            if temp_node_gpu[j] == 1: #只放在用过的gpu里
                                continue
                            item = temp_node_gpu[j]
                            if task_gpu.sum() < item:
                                temp_old = self.state_only_float[float_to_int(item)]
                                temp_new = self.state_only_float[float_to_int(item - task_gpu.sum())]
                                if temp < temp_old - temp_new:
                                    temp = temp_old - temp_new
                                    select = j
                        if select != -1:
                            temp_old = state_old + self.state_only_float[float_to_int(temp_node_gpu[select])]
                            temp_new = self.state_int[int(temp_node_cpu - task_cpu)][gpu_int_old] + \
                                       self.state_only_float[float_to_int(temp_node_gpu[select] - task_gpu.sum())]
                            if now_priority < temp_old - temp_new:
                                now_priority = temp_old - temp_new
                                now_select = node
                                temp_select[select] = 0
                    else:
                        max_float_site = get_max_float(temp_node_gpu)
                        state_old = self.state_float[int(temp_node_cpu)][float_to_int(temp_node_gpu[max_float_site])]
                        temp = -1e8
                        for j in range(len(temp_node_gpu)):
                            if temp_node_gpu[j] == 1:
                                continue
                            if j == max_float_site:
                                continue
                            item = temp_node_gpu[j]
                            temp_old = 0
                            temp_new = 0
                            if task_gpu.sum() < item:
                                temp_old = self.state_only_float[float_to_int(item)]
                                temp_new = self.state_only_float[float_to_int(item - task_gpu.sum())]
                            if temp < temp_old - temp_new:
                                temp = temp_old - temp_new
                                select = j
                        if select != -1:
                            temp_old = state_old + self.state_only_float[float_to_int(temp_node_gpu[select])]
                            temp_new = self.state_float[int(temp_node_cpu - task_cpu)][
                                           float_to_int(temp_node_gpu[max_float_site])] + self.state_only_float[
                                           float_to_int(temp_node_gpu[select] - task_gpu.sum())]
                            if now_priority < temp_old - temp_new:
                                now_priority = temp_old - temp_new
                                now_select = node
                                temp_select[select] = 0

                        if temp_node_gpu[max_float_site] > task_gpu.sum():
                            if gpu_float_old == 1 or temp_node_gpu[max_float_site] - task_gpu.sum() > \
                                    temp_node_gpu[get_second_float(temp_node_gpu)]:
                                temp_old = state_old
                                temp_new = self.state_float[int(temp_node_cpu - task_cpu)][
                                    float_to_int(temp_node_gpu[max_float_site] - task_gpu.sum())]
                            else:
                                temp_old = state_old + self.state_only_float[float_to_int(temp_node_gpu[get_second_float(temp_node_gpu)])]
                                temp_new = self.state_float[int(temp_node_cpu - task_cpu)][
                                               float_to_int(temp_node_gpu[get_second_float(temp_node_gpu)])] + self.state_only_float[
                                               float_to_int(temp_node_gpu[max_float_site] - task_gpu.sum())]
                            if now_priority < temp_old - temp_new:
                                now_priority = temp_old - temp_new
                                now_select = node
                                temp_select[get_second_float(temp_node_gpu)] = 0
                if select != -1:
                    gpu_select = temp_select
                    continue

                max_float_site = get_max_float(temp_node_gpu)
                if gpu_int_old >= task_gpu.sum():
                    state_old = self.state_int[int(temp_node_cpu)][int(gpu_int_old)]
                    if gpu_float_old > 0 and (gpu_int_old - task_gpu.sum()) < temp_node_gpu[max_float_site]:
                        temp_old = state_old + self.state_only_float[float_to_int(temp_node_gpu[max_float_site])]
                        temp_new = self.state_float[int(temp_node_cpu - task_cpu)][
                                       float_to_int(temp_node_gpu[max_float_site])] + self.state_only_float[
                                       float_to_int(gpu_int_old - task_gpu.sum())]
                    elif (gpu_int_old - task_gpu.sum()) < 0:
                        temp_old = state_old
                        temp_new = self.state_float[int(temp_node_cpu - task_cpu)][
                            float_to_int(gpu_int_old - task_gpu.sum())]
                    else:
                        temp_old = state_old
                        temp_new = self.state_int[int(temp_node_cpu - task_cpu)][
                                       int(gpu_int_old - task_gpu.sum())] + self.state_only_float[
                                       float_to_int(gpu_int_old - task_gpu.sum())]
                    if now_priority < temp_old - temp_new:
                        now_priority = temp_old - temp_new
                        now_select = node
                        for i in range(len(task_gpu)):
                            for j in range(len(temp_node_gpu)):
                                if np.any(task_gpu[i] > temp_node_gpu[j]):
                                    continue
                                else:
                                    temp_select[j] = i
                                    temp_node_gpu[j] -= task_gpu[i]
                                    break
                        assert len(temp_select) == len(task_gpu),"len(temp_select) != len(task_gpu)"
                        gpu_select = temp_select
        if now_select != -1:
            self.set_task(now_select, task, gpu_select)
            return True
        else:
            return False


def get_max_float(x):
    temp = -1
    select = -1
    for i in range(len(x)):
        if x[i] == 1:
            continue
        if x[i] > temp:
            temp = x[i]
            select = i
    return select

def get_second_float(x):
    temp = -1
    select = -1
    max_site = get_max_float(x)
    for i in range(len(x)):
        if x[i] == 1:
            continue
        if i == max_site:
            continue
        if x[i] > temp:
            temp = x[i]
            select = i
    return select

def float_to_int(x):
    return int(x * 10) % 10


    #             if task_gpu.sum() < 1:  # gpu使用量为小数
    #                 for j in range(len(temp_node_gpu)):
    #                     if np.any(task_gpu[0] > temp_node_gpu[j]):
    #                         continue
    #                     else:
    #                         temp_node_gpu[j] -= task_gpu[0]
    #                         temp_priority = self.get_state_value(temp_node_cpu_old, temp_node_gpu_old,
    #                                                              temp_node_cpu_new, temp_node_gpu)
    #                         temp_node_gpu[j] += task_gpu[0]
    #                         if now_priority < temp_priority:
    #                             now_priority = temp_priority
    #                             gpu_select = j
    #                 if gpu_select != -1:
    #                     now_select = node
    #                     temp_select[gpu_select] = 0
    #                     temp_node_gpu[gpu_select] -= task_gpu[0]
    #             else:  # gpu使用量为整数
    #                 temp_gpu_select = -1
    #                 for i in range(len(task_gpu)):
    #                     for j in range(len(temp_node_gpu)):
    #                         if np.any(task_gpu[i] > temp_node_gpu[j]):
    #                             continue
    #                         else:
    #                             temp_select[j] = i
    #                             temp_node_gpu[j] -= task_gpu[i]
    #                 if len(temp_select) == len(task_gpu):
    #                     temp_priority = self.get_state_value(temp_node_cpu_old, temp_node_gpu_old, temp_node_cpu_new,
    #                                                          temp_node_gpu)
    #                     if now_priority < temp_priority:
    #                         now_priority = temp_priority
    #                         now_select = node
    #                         gpu_select = temp_select
    #     if now_select != -1:
    #         self.set_task(now_select, task, gpu_select)
    #         return True
    #     else:
    #         return False
    #
    # def get_state_value(self, temp_node_cpu_old, temp_node_gpu_old, temp_node_cpu_new, temp_node_gpu):
    #     gpu_int_old = np.count_nonzero(temp_node_gpu_old == 1)
    #     gpu_float_old = np.count_nonzero(temp_node_gpu_old > __zero__) - gpu_int_old
    #     gpu_int_new = np.count_nonzero(temp_node_gpu == 1)
    #     gpu_float_new = np.count_nonzero(temp_node_gpu > __zero__) - gpu_int_new
    #     if gpu_int_old > gpu_int_new:
    #         if gpu_float_old == gpu_float_new:
    #             return self.state_int[temp_node_cpu_old][gpu_int_old] - self.state_int[[temp_node_cpu_old]][gpu_int_old]
    #         else:
    #             pass
    #     return 0
