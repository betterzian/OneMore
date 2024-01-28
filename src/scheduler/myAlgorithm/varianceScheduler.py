from src.scheduler.schedulerClass import Scheduler
import numpy as np
class VarianceScheduler(Scheduler):
    def __init__(self,cluster,can_predict = True):
        super().__init__(cluster,can_predict)

    def run(self,task):
        cpu,gpu = self.get_task_info(task)
        now_var = 99999999.9
        second_var = 99999999.9
        now_add = 0
        now_select = -1
        second_select = -1
        for node in self.cluster:
            temp_node_cpu,_ = self.get_node_info(node)
            temp = cpu - temp_node_cpu
            condition = temp > 0
            if np.any(condition):
                address = np.where(condition)[0][0]
                if now_add < address:
                    now_add = address
                    second_var = np.var(temp_node_cpu[:address] - cpu[:address]) / address
                    second_select = node
                elif now_add == address and address > 0:
                    temp_var = np.var(temp_node_cpu[:address] - cpu[:address]) / address
                    if temp_var < second_var:
                        second_var = temp_var
                        second_select = node
                continue
            else:
                temp_var = np.var(temp_node_cpu - cpu)
            if temp_var < now_var:
                now_var = temp_var
                now_select = node
        if now_select != -1:
            self.set_task(now_select,task)
        elif second_select != -1:
            self.set_task(second_select,task)