from src.scheduler.schedulerClass import Scheduler
from src.scheduler.myAlgorithm.ResourceShaping import ResourceShaping
from src.scheduler.myAlgorithm.DGGSVbyNet import DGGSVbyNet

class OneMore(Scheduler):
    def __init__(self, cluster, can_predict=True, task_mem={}, node_mem={}):
        super().__init__(cluster, can_predict, task_mem, node_mem)
        self.online_scheduler = ResourceShaping(cluster, can_predict, self._task_mem, self._node_mem)
        self.offline_scheduler = DGGSVbyNet(cluster, can_predict, self._task_mem, self._node_mem)

    def run(self, task):
        if task.get_arrive_time() < 0:
            return self.online_scheduler.run(task)
        else:
            return self.offline_scheduler.run(task)

    def init_again(self):
        self.offline_scheduler.init_again()

    def release(self):
        self.offline_scheduler.release()

    def get_reschedule_num(self):
        return self._reschedule_num + self.online_scheduler.get_reschedule_num() + self.offline_scheduler.get_reschedule_num()

    def get_fail_num(self):
        return self._fail_num + self.online_scheduler.get_fail_num() + self.offline_scheduler.get_fail_num()

    def get_task_len(self):
        return self._task_len + self.offline_scheduler.get_task_len() + self.online_scheduler.get_task_len()

    def get_node_cache_num(self):
        return self._node_cache_num + self.online_scheduler.get_node_cache_num() + self.offline_scheduler.get_node_cache_num()

    def get_task_cache_num(self):
        return self._task_cache_num + self.offline_scheduler.get_task_cache_num() + self.online_scheduler.get_task_cache_num()

    def get_node_no_cache_num(self):
        return self._node_no_cache_num + self.online_scheduler.get_node_no_cache_num() + self.offline_scheduler.get_node_no_cache_num()

    def get_task_no_cache_num(self):
        return self._task_no_cache_num + self.offline_scheduler.get_task_no_cache_num() + self.online_scheduler.get_task_no_cache_num()

    def get_force_num(self):
        return self._force_num + self.offline_scheduler.get_force_num() + self.online_scheduler.get_force_num()
