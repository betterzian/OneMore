from src.scheduler.schedulerClass import Scheduler
import numpy as np
class FirstFit(Scheduler):
    def __init__(self,cluster,can_predict = True,task_mem = {},node_mem = {}):
        super().__init__(cluster,can_predict,task_mem,node_mem)

    def run(self,task):
        task_cpu, task_gpu = self.get_task_info(task)
        now_select = -1
        gpu_select = {}
        for node in self.cluster:
            temp_select = {}
            temp_node_cpu, temp_node_gpu = self.get_node_info(node)
            if np.any(task_cpu > temp_node_cpu):
                continue
            else:
                for i in range(len(task_gpu)):
                    for j in range(len(temp_node_gpu)):
                        if np.any(task_gpu[i] > temp_node_gpu[j]):
                            continue
                        else:
                            temp_select[j] = i
                            temp_node_gpu[j] -= task_gpu[i]
                            break
            if len(temp_select) == len(task_gpu):
                now_select = node
                gpu_select = temp_select
                break

        if now_select != -1:
            self.set_task(now_select, task, gpu_select)
            return True
        else:
            return False