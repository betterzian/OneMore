from src.scheduler.schedulerClass import Scheduler
from src.scheduler.myAlgorithm.varianceScheduler import VarianceScheduler
from src.scheduler.myAlgorithm.sssppScheduler import SSSPPScheduler
class OneMoreScheduler(Scheduler):
    def __init__(self, cluster, can_predict=True):
        super().__init__(cluster, can_predict)
        self.online_scheduler = VarianceScheduler(cluster,can_predict)
        self.offline_scheduler = SSSPPScheduler(cluster,can_predict)

    def run(self,task):
        if task.get_arrive_time() < 0:
            return self.online_scheduler.run(task)
        else:
            return self.offline_scheduler.run(task)