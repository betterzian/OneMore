from src.scheduler.schedulerClass import Scheduler
from src.scheduler.myAlgorithm.varianceScheduler import VarianceScheduler
from src.scheduler.myAlgorithm.modelScheduler import ModelScheduler
from src.envSim.timeSim import TimeHolder
class OneMoreModelScheduler(Scheduler):
    def __init__(self,cluster,can_predict = True,task_mem = {},node_mem = {}):
        super().__init__(cluster,can_predict,task_mem,node_mem)
        self.online_scheduler = VarianceScheduler(cluster, can_predict,self._task_mem,self._node_mem)
        self.offline_scheduler = ModelScheduler(cluster, can_predict,self._task_mem,self._node_mem)
    def run(self,task):
        if task.get_arrive_time() < 0:
            return self.online_scheduler.run(task)
        else:
            return self.offline_scheduler.run(task)
