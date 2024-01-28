from src.scheduler.schedulerClass import Scheduler
import numpy as np
class FirstFitScheduler(Scheduler):
    def __init__(self,cluster,can_predict = False):
        super().__init__(cluster,can_predict)

    def run(self,task):
        cpu,gpu = self.get_task_info(task)
        now_priority = 99999999.0
        now_select = -1
        for node in self.cluster:
            temp_node_cpu,_ = self.get_node_info(node)
            if np.any(cpu > temp_node_cpu):
                continue
            else:
                temp_priority = np.sum(temp_node_cpu - cpu)
            if temp_priority < now_priority:
                now_priority = temp_priority
                now_select = node
        if now_select != -1:
            self.set_task(now_select,task)