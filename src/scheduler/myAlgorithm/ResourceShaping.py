from src.scheduler.schedulerClass import Scheduler
import numpy as np


class ResourceShaping(Scheduler):
    def __init__(self, cluster, can_predict=True, task_mem={}, node_mem={}):
        super().__init__(cluster, can_predict, task_mem, node_mem)

    def run(self, task):
        task_cpu, task_gpu = self.get_task_info(task)
        now_priority = -99999999999.0
        now_select = -1
        gpu_select = {}
        for node in self.cluster:
            temp_select = {}
            temp_node_cpu, temp_node_gpu = self.get_node_info(node)
            if np.any(task_cpu > temp_node_cpu):
                continue
            else:
                old_priority = temp_node_cpu
                old_priority = np.concatenate((old_priority, np.sum(temp_node_gpu, axis=0) * self._rate), axis=0)
                temp_priority = temp_node_cpu - task_cpu
                for i in range(len(task_gpu)):
                    temp_gpu_select = -1
                    gpu_priority = -9999999999.0
                    for j in range(len(temp_node_gpu)):
                        if np.any(task_gpu[i] > temp_node_gpu[j]):
                            continue
                        else:
                            temp_gpu_priority = np.var(temp_node_gpu[j]) - np.var(temp_node_gpu[j] - task_gpu[i])
                            if temp_gpu_priority > gpu_priority:
                                gpu_priority = temp_gpu_priority
                                temp_gpu_select = j
                    if temp_gpu_select != -1:
                        temp_select[temp_gpu_select] = i
                        temp_node_gpu[temp_gpu_select] -= task_gpu[i]
                if len(temp_select) == len(task_gpu):
                    temp_priority = np.concatenate((temp_priority, np.sum(temp_node_gpu, axis=0) * self._rate), axis=0)
                    temp_priority = np.var(old_priority) - np.var(temp_priority)
                    if temp_priority > now_priority:
                        now_priority = temp_priority
                        now_select = node
                        gpu_select = temp_select
        if now_select != -1:
            self.set_task(now_select, task, gpu_select)
            return True
        else:
            return False