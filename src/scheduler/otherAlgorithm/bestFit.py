from src.scheduler.schedulerClass import Scheduler
import numpy as np

class BestFit(Scheduler):
    def __init__(self,cluster,can_predict = True,task_mem = {},node_mem = {}):
        super().__init__(cluster,can_predict,task_mem,node_mem)

    def run(self,task):
        task_cpu, task_gpu = self.get_task_info(task)
        now_priority = 999999999.0
        now_select = -1
        gpu_select = {}
        for node in self.cluster:
            temp_select = {}
            temp_node_cpu, temp_node_gpu = self.get_node_info(node)
            if np.any(task_cpu > temp_node_cpu):
                continue
            else:
                if len(task_gpu) == 0:
                    task_gpu_sum = 0
                else:
                    task_gpu_sum = task_gpu.sum()
                temp_priority = temp_node_cpu.sum() - task_cpu.sum() + self._rate * (temp_node_gpu.sum() - task_gpu_sum)
                for i in range(len(task_gpu)):
                    gpu_priority = 999999999.0
                    temp_gpu_select = -1
                    for j in range(len(temp_node_gpu)):
                        if np.any(task_gpu[i] > temp_node_gpu[j]):
                            continue
                        else:
                            temp_gpu_priority = np.sum(temp_node_gpu[j] - task_gpu[i])
                            if temp_gpu_priority < gpu_priority:
                                gpu_priority = temp_gpu_priority
                                temp_gpu_select = j
                    if temp_gpu_select != -1:
                        temp_select[temp_gpu_select] = i
                        temp_node_gpu[temp_gpu_select] -= task_gpu[i]
                if len(temp_select) == len(task_gpu):
                    if temp_priority < now_priority:
                        now_select = node
                        gpu_select = temp_select
        if now_select != -1:
            self.set_task(now_select, task, gpu_select)
            return True
        else:
            return False