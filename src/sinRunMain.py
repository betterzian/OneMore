import sys
sys.path.append("../")
import argparse
from multiprocessing import Pool
from datetime import datetime
from src.envSim.simRun import sim_run
from src.envSim.timeSim import TimeHolder
from src.envSim.simParam import ParamHolder
from src.envSim.offlineTaskDataProcess import offline_data_process
from envSim.generateEnv import generate_cluster, generate_online_task_list, generate_offline_task_list
from scheduler.schedulerList import init_scheduler
from src.envSim.TXTtoCSV import txt_to_csv


def run(args_dict):
    ParamHolder().init_again(args_dict)
    TimeHolder().init_again()
    offline_data_process(ParamHolder().filename)
    args_dict["cgr"] = ParamHolder().cpu_gpu_rate
    args_dict["avgsize"] = ParamHolder().avgsize
    cluster = generate_cluster()
    online_task_list = generate_online_task_list()
    offline_task_list = generate_offline_task_list()
    schedulers = []
    schedulers.extend(init_scheduler(cluster, False))
    schedulers.extend(init_scheduler(cluster, True))
    if not args_dict["test"]:
        p = Pool(len(schedulers))
        for scheduler in schedulers:
            p.apply_async(sim_run, args=(scheduler, online_task_list, offline_task_list, args_dict))
        p.close()
        p.join()
    else:
        for scheduler in schedulers:
            print("单进程,测试用")
            sim_run(scheduler, online_task_list, offline_task_list, args_dict)
    txt_to_csv(ParamHolder().csv_name)


# 单次模拟运行
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='param')
    parser.add_argument('-test', action='store_true')
    parser.add_argument('-cuda', type=int, default=0)
    parser.add_argument('-t', type=int, default=8640)
    parser.add_argument('-tcp', type=int, default=8640)
    parser.add_argument('-tap', type=int, default=30)
    parser.add_argument('-ontn', type=int, default=10000)
    parser.add_argument('-oftn', type=int, default=1400)
    parser.add_argument('-filename', type=str, default="openb_pod_list_multigpu50")
    # 暂未用到的可换参数:
    parser.add_argument('-tif', type=int, default=0)
    parser.add_argument('-nt', type=list, default=((64, 2), (64, 8), (96, 4), (96, 8), (104, 2), (128, 1), (128, 8)))
    parser.add_argument('-nn', type=tuple, default=((20, 20, 20, 20, 20, 20, 20)))
    parser.add_argument('-tbs', type=int, default=90)
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict["tl"] = args_dict["tif"] + args_dict["t"] * 2
    args_dict["tef"] = args_dict["tif"] + args_dict["t"]
    if args_dict["cuda"] == -1:
        args_dict["device"] = "cpu"
    else:
        args_dict["device"] = "cuda"
    args_dict["cuda"] = str(args_dict["cuda"])
    current_time = datetime.now()
    args_dict["csv_name"] = str(current_time.year) + "_" + str(current_time.month) + "_" + str(
        current_time.day) + "_" + str(
        current_time.hour) + "_" + str(current_time.minute) + "_" + str(current_time.second)
    args_dict["avgsize"] = 1
    run(args_dict)
