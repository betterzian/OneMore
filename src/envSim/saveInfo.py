from src.scheduler.schedulerClass import Scheduler
import pandas as pd
import csv
from src.simParam import args


def save_info(scheduler: Scheduler):
    cluster = scheduler.cluster
    args_dict = {}
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

    args_dict["cpu_rate"] = (max_cpu - rest_cpu) * 1.0 / max_cpu
    args_dict["gpu_rate"] = (max_gpu - rest_gpu) * 1.0 / max_gpu
    args_dict["reschedule_num"] = scheduler.reschedule_num
    args_dict["fail_num"] = scheduler.fail_num
    args_dict["success_num"] = success_num
    args_dict["task_cache_num"] = scheduler.task_cache_num
    args_dict["task_no_cache_num"] = scheduler.task_no_cache_num
    args_dict["node_cache_num"] = scheduler.node_cache_num
    args_dict["node_no_cache_num"] = scheduler.node_no_cache_num
    args_dict["scheduler_name"] = type(scheduler).__name__
    args_dict["can_predict"] = scheduler.get_can_predict()
    args_dict["run_time"] = scheduler.get_time()

    args_dict["time_len"] = args["tl"]
    args_dict["time_init_flag"] = args["tif"]
    args_dict["time_can_predict"] = args["tcp"]
    args_dict["time_block_size"] = args["tbs"]
    args_dict["time_accurately_predict"] = args["tap"]
    args_dict["online_task_num"] = args["ontn"]
    args_dict["offline_task_num"] = args["oftn"]
    args_dict["cpu_gpu_rate"] = args["cgr"]
    args_dict["node_type"] = [args["nt"]]
    args_dict["node_num"] = [args["nn"]]

    str_list = pd.DataFrame.from_dict(args_dict)
    str_list.to_csv('../output/scheduler_result.txt', index=False, header=True, sep='\t', mode='a', encoding="utf-8",
                    quoting=csv.QUOTE_NONE, escapechar=',')
