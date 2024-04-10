import csv
import pandas as pd
from src.envSim.timeSim import TimeHolder
from src.envSim.simParam import ParamHolder
from src.scheduler.schedulerClass import Scheduler


def save_info(scheduler: Scheduler):
    cluster = scheduler.cluster
    args_dict = {}
    max_cpu = 0
    max_gpu = 0
    rest_cpu = 0
    rest_gpu = 0
    offline_task_cpu = 0
    offline_task_gpu = 0
    success_num = 0
    node_dict = {}
    length = TimeHolder().get_time_end_flag() - TimeHolder().get_time_init_flag()
    for node in cluster:
        max_cpu += node.get_max_cpu()[:length].sum()
        max_gpu += node.get_max_gpu()[:,:length].sum()
        rest_cpu += node.get_cpu_info()[:length].sum()
        rest_gpu += node.get_gpu_info()[:,:length].sum()
        temp_success_num,temp_offline_task_cpu,temp_offline_task_gpu = node.get_success_num()
        success_num += temp_success_num
        offline_task_cpu += temp_offline_task_cpu
        offline_task_gpu += temp_offline_task_gpu
        node_dict[node.get_id()] = [len(node.get_online_task()),node.get_cpu_info().max(),node.get_cpu_info().min(),node.get_max_gpu().max(),node.get_cpu_info(),node.get_max_gpu()]
    args_dict["cpu_rate"] = (max_cpu - rest_cpu) * 1.0 / max_cpu
    args_dict["gpu_rate"] = (max_gpu - rest_gpu) * 1.0 / max_gpu
    args_dict["reschedule_num"] = scheduler.get_reschedule_num()
    args_dict["success_num"] = success_num
    args_dict["force_num"] = scheduler.get_force_num()
    args_dict["scheduler_name"] = type(scheduler).__name__
    args_dict["can_predict"] = scheduler.get_can_predict()
    args_dict["run_time"] = scheduler.get_time()
    args_dict["time_can_predict"] = ParamHolder().time_can_predict
    args_dict["time_accurately_predict"] = ParamHolder().time_accurately_predict
    args_dict["online_task_num"] = ParamHolder().online_task_num
    args_dict["offline_task_num"] = ParamHolder().offline_task_num
    args_dict["node_type"] = [ParamHolder().node_type]
    args_dict["node_num"] = [ParamHolder().node_num]
    args_dict["cpu_gpu_rate"] = [ParamHolder().cpu_gpu_rate]
    args_dict["filename"] = ParamHolder().filename

    #Param 校准
    args_dict["prob"] = ParamHolder().prob * 1.0 / 100
    args_dict["weight"] = ParamHolder().weight
    args_dict["avgsize"] = ParamHolder().avgsize
    args_dict["weight_rate"] = ParamHolder().weight_rate

    str_list = pd.DataFrame.from_dict(args_dict)
    str_list.to_csv('../output/scheduler_result_'+str(ParamHolder().csv_name)+'.txt', index=False, header=True, sep='\t', mode='a', encoding="utf-8",quoting=csv.QUOTE_NONE, escapechar=',')
