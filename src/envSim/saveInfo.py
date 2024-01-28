from src.scheduler.schedulerClass import Scheduler
import pandas as pd
import csv
def save_info(scheduler:Scheduler):
    cluster = scheduler.cluster
    max_cpu = 0
    max_gpu = 0
    rest_cpu = 0
    rest_gpu = 0
    success_num = 0
    for node in cluster:
        max_cpu += node.get_max_cpu().sum()
        max_gpu += node.get_max_gpu().sum()
        rest_cpu += node.get_cpu_info().sum()
        rest_gpu += node.get_gpu_info().sum()
        success_num += node.get_success_num()
    cpu_rate = (max_cpu-rest_cpu)*1.0/max_cpu
    gpu_rate = (max_gpu-rest_gpu)*1.0/max_gpu
    str_list = [cpu_rate,gpu_rate,scheduler.reschedule_num,scheduler.fail_num,success_num,scheduler.task_cache_num,scheduler.task_no_cache_num,scheduler.node_cache_num,scheduler.node_no_cache_num,type(scheduler).__name__,scheduler.get_can_predict()]
    str_list= pd.DataFrame(str_list)
    str_list.to_csv('../../output/scheduler_result.txt', index=False, header=False, sep='\t', mode='a',encoding="utf-8", quoting=csv.QUOTE_NONE, escapechar=',')
    print(str_list)