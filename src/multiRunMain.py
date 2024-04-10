import sys
sys.path.append("../")
import random
import argparse
import torch.multiprocessing as mp
from copy import deepcopy
from datetime import datetime
from src.config import args_dict_param, args_dict_compare
from src.envSim.TXTtoCSV import txt_to_csv
from src.envSim.simRun import sim_run
from src.envSim.simParam import ParamHolder
from src.envSim.timeSim import TimeHolder
from src.envSim.offlineTaskDataProcess import offline_data_process
from src.envSim.generateEnv import generate_cluster, generate_online_task_list, generate_offline_task_list
from src.scheduler.schedulerList import init_scheduler


ready_list = []


def dummy_func(count):
    global ready_list
    ready_list.append(count)


def generate(args_dict):
    ParamHolder().init_again(args_dict)
    TimeHolder().init_again()
    offline_data_process(ParamHolder().filename)
    args_dict["cgr"] = ParamHolder().cpu_gpu_rate
    args_dict["avgsize"] = ParamHolder().avgsize
    args_dict["zero_task"] = ParamHolder().zero_task
    cluster = generate_cluster()
    online_task_list = generate_online_task_list()
    offline_task_list = generate_offline_task_list()
    return args_dict, cluster, online_task_list, offline_task_list


def run_scheduler(args, args_dict, schedulers, result, count, p, online_task_list, offline_task_list):
    if not args.test:
        for scheduler in schedulers:
            result[count] = p.apply_async(sim_run, args=(
                scheduler, online_task_list, offline_task_list, deepcopy(args_dict), count),
                                          callback=dummy_func)
            count += 1
            for ready in ready_list:
                result[ready].wait()
                del result[ready]
            ready_list.clear()
    else:
        for scheduler in schedulers:
            print("单进程,测试用")
            sim_run(scheduler, online_task_list, offline_task_list, args_dict, count)
    return count


def process(args, args_dict, result, count, p):
    args_dict, cluster, online_task_list, offline_task_list = generate(args_dict)
    if args.param:
        schedulers = init_scheduler(cluster, False)
        count = run_scheduler(args, args_dict, schedulers, result, count, p, online_task_list, offline_task_list)
    else:
        ok = False
        for tcp in args_dict["tcp_list"]:
            args_dict["tcp"] = tcp
            for tap in args_dict["tap_list"]:
                args_dict["tap"] = tap
                if len(args_dict["src_cuda"]) == 0:
                    args_dict["src_cuda"] = [i for i in range(args.mc)]
                    random.shuffle(args_dict["src_cuda"])
                args_dict["cuda"] = args_dict["src_cuda"].pop()
                ParamHolder().init_again(args_dict)
                TimeHolder().init_again()
                if not ok:
                    schedulers = []
                    schedulers.extend(init_scheduler(cluster, False))
                    schedulers.extend(init_scheduler(cluster, True))
                    ok = True
                else:
                    schedulers = init_scheduler(cluster,True)
                count = run_scheduler(args, args_dict, schedulers, result, count, p, online_task_list, offline_task_list)
    return count


def run(args):
    global ready_list
    current_time = datetime.now()
    csvname = str(current_time.year) + "_" + str(current_time.month) + "_" + str(
        current_time.day) + "_" + str(current_time.hour) + "_" + str(current_time.minute) + "_" + str(
        current_time.second)
    p = mp.Pool(args.pool)
    result = {}
    count = 0
    for i in range(args.ins):
        if args.param:
            csv_name = "all_param_" + csvname
            args_dict = {"tl": 2, "tif": 0, "tef": 1, "tbs": 1, "nt": ((32, 32), (96, 96)),
                         "nn": ((20, 20)), "tap": 1, "tcp": 1, "csv_name": csv_name,
                         "device": "cpu", "ontn": 0, "oftn": 0, "cuda": 1, "avgsize": 1,}
            args_dict.update(args_dict_param)
            for prob in args_dict["prob_list"]:
                args_dict["prob"] = prob
                test_count = 0
                for size in args_dict["size_list"]:
                    for small_task in args_dict["small_task_list"]:
                        from src.envSim.generateEnv import generate_task_prob
                        test_count += 1
                        args_dict["filename"] = f"param_{test_count}"
                        ParamHolder().init_again(args_dict)
                        generate_task_prob(min=0, max=20, size=(size, size), small_task=(small_task, small_task))
                        for task_num in args_dict["task_num_list"]:
                            args_dict["oftn"] = task_num
                            for node_count in args_dict["node_count_list"]:
                                args_dict["nn"] = ((node_count, node_count))
                                count = process(args, args_dict, result, count, p)
        else:
            csv_name = "all_compare_" + csvname
            args_dict = {"tl": 17280, "tif": 0, "tef": 8640, "tbs": 90,
                         "nt": ((64, 2), (64, 8), (96, 4), (96, 8), (104, 2), (128, 1), (128, 8)),
                         "nn": ((20, 20, 20, 20, 20, 20, 20)), "csv_name": csv_name,
                         "device": "gpu", "cuda": 1, "avgsize": 1, "weight_rate": 1, "weight": 1}
            args_dict.update(args_dict_compare)
            args_dict["src_cuda"] = [i for i in range(args.mc)]
            random.shuffle(args_dict["src_cuda"])
            for filename in args_dict["filename_list"]:
                args_dict["filename"] = filename
                for oftn in args_dict["oftn_ontn_list"].keys():
                    args_dict["oftn"] = oftn
                    for ontn in args_dict["oftn_ontn_list"][oftn]:
                        args_dict["ontn"] = ontn
                        count = process(args, args_dict, result, count, p)
    p.close()
    p.join()
    txt_to_csv(csv_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='param')
    parser.add_argument('-test', action='store_true')
    parser.add_argument('-param', action='store_true')
    parser.add_argument('-pool', type=int, default=64)
    parser.add_argument('-mc', type=int, default=4)
    parser.add_argument('-ins', type=int, default=5)
    args = parser.parse_args()
    run(args)
