from src.scheduler.schedulerClass import Scheduler
import numpy as np
class LargeGreedyScheduler(Scheduler):
    def __init__(self,cluster,can_predict = True):
        super().__init__(cluster,can_predict)


    def run(self,task):
        cpu,gpu = self.get_task_info(task)
        nowPriority = -1
        now_select = -1
        for node in self.cluster:
            temp_node_cpu,_ = self.get_node_info(node)
            if np.any(cpu > temp_node_cpu):
                continue
            else:
                tempPriority = np.sum(temp_node_cpu - cpu)
            if tempPriority > nowPriority:
                nowPriority = tempPriority
                now_select = node
        if now_select != -1:
            self.set_task(now_select,task)
