from src.scheduler.schedulerClass import Scheduler
from src.envSim.simParam import ParamHolder
import numpy as np

class GGSV(Scheduler):
    def __init__(self,cluster, task_mem = {},node_mem = {}):
        super().__init__(cluster,False,task_mem,node_mem)
        self.__state = np.loadtxt(f"../srcData/state_value/{ParamHolder().filename}/state{ParamHolder().prob}.csv", delimiter=",")
        self.__prob = ParamHolder().prob * 1.0 / 100
    def run(self, task):
        task_cpu, task_gpu = self.get_task_info(task)
        task_state = np.array([task_cpu.sum(), task_gpu.sum()])
        now_priority = -1e8
        now_select = -1
        for node in self.cluster:
            temp_node_cpu, temp_node_gpu = self.get_node_info(node)
            old_state = np.array([temp_node_cpu.sum(), temp_node_gpu.sum()])

            if np.any(task_state > old_state):
                continue
            else:
                new_state = old_state - task_state
                temp_priority = self.__state[int(old_state[0])][int(old_state[1])] - self.__state[int(new_state[0])][int(new_state[1])]
                if now_priority < temp_priority:
                    now_priority = temp_priority
                    now_select = node
        if now_select != -1:
            self.set_task(now_select, task, {0:0})
            return True
        else:
            return False