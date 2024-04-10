from src.scheduler.schedulerClass import Scheduler
from src.envSim.generateEnv import generate_offline_task_list
import numpy as np

class FGD(Scheduler):
    def __init__(self, cluster, can_predict=True, task_mem={}, node_mem={}):
        super().__init__(cluster, can_predict, task_mem, node_mem)
        self.__offline_task_list = generate_offline_task_list(all_bool=True)
        self.__offline_task_len = len(self.__offline_task_list)
        #self.online_scheduler = VarianceScheduler(cluster, can_predict, self._task_mem, self._node_mem)

    def run(self, task):
        # if task.get_arrive_time() < 0:
        #     return self.online_scheduler.run(task)
        task_cpu, task_gpu = self.get_task_info(task)
        if len(task_gpu) > 0:
            task_gpu = task_gpu[:, 0]
        now_priority = 1e8
        now_select = -1
        gpu_select = {}
        for node in self.cluster:
            temp_select = {}
            temp_node_cpu, temp_node_gpu = self.get_node_info(node)
            temp_node_gpu = temp_node_gpu[:, 0]
            if np.any(task_cpu > temp_node_cpu):
                continue
            else:
                old_value = self.__get_frag_num(temp_node_cpu.min(),temp_node_gpu)
                temp_node_cpu = (temp_node_cpu - task_cpu).min()
                if 0 < task_gpu.sum() < 1:
                    for i in range(len(temp_node_gpu)):
                        if task_gpu.sum() <= temp_node_gpu[i]:
                            temp = temp_node_gpu - 0
                            temp[i] -= task_gpu.sum()
                            new_value = self.__get_frag_num(temp_node_cpu,temp)
                            if new_value - old_value < now_priority:
                                now_priority = new_value - old_value
                                now_select = node
                                gpu_select = {i:0}
                else:
                    for i in range(len(task_gpu)):
                        for j in range(len(temp_node_gpu)):
                            if task_gpu[i] > temp_node_gpu[j]:
                                continue
                            else:
                                temp_select[j] = i
                                temp_node_gpu[j] -= task_gpu[i]
                                break
                    if len(task_gpu) == len(temp_select):
                        new_value = self.__get_frag_num(temp_node_cpu,temp_node_gpu)
                        if new_value - old_value < now_priority:
                            now_priority = new_value - old_value
                            now_select = node
                            gpu_select = temp_select
        if now_select != -1:
            self.set_task(now_select, task, gpu_select)
            return True
        else:
            return False

    def __get_frag_num(self,node_cpu,node_gpu):
        gpu_sum = np.count_nonzero(node_gpu > 0)
        rule1 = self.__offline_task_list[:,0] > node_cpu.min()
        rule2 = self.__offline_task_list[:,1] > node_gpu.max()
        indices = self.__offline_task_list[:,1] <= 1
        rule2 = rule2 & indices
        rule3 = self.__offline_task_list[:,1] > np.count_nonzero(node_gpu == 1)
        rule3 = rule3 & ~indices
        rule4 = self.__offline_task_list[:,1] == 0
        case1andcase3 = rule1 | rule2 | rule3 | rule4
        count = np.count_nonzero(case1andcase3)* gpu_sum
        rule4 = ~ case1andcase3
        for i in node_gpu:
            if i == 0:
                continue
            rule = self.__offline_task_list[:,1] > i
            count += np.count_nonzero(rule4 & rule)
        return count * 1.0 / self.__offline_task_len

