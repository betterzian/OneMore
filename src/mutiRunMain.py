import sys
sys.path.append("../")
import torch
from datetime import datetime
import torch.multiprocessing as mp
from src.envSim.TXTtoCSV import txt_to_csv
from src.simRun import sim_run
from copy import deepcopy

def init_Holders(args_dict):
    from src.envSim.simParam import ParamHolder
    ParamHolder(args_dict)
    from src.envSim.timeSim import TimeHolder
    TimeHolder()


def callback(result):
    print("Task completed with result:", result)


def run():
    from src.envSim.simParam import ParamHolder
    ParamHolder()
    from src.envSim.timeSim import TimeHolder
    TimeHolder()
    current_time = datetime.now()
    csv_name = str(current_time.year) + "_" + str(current_time.month) + "_" + str(
        current_time.day) + "_" + str(current_time.hour) + "_" + str(current_time.minute) + "_" + str(
        current_time.second)
    tcp_list = [1080, 2160, 4320, 8640]
    tap_list = [30, 90, 180, 360]
    ontn_list = {1000: [1400, 1600, 1800], 3000: [1400, 1600, 1800], 5000: [1400, 1600, 1800]}
    filename_list = ["openb_pod_list_multigpu50"]
    p = mp.Pool(50)
    args_dict = {"test": False,
                 "tl": 17280,
                 "tif": 0,
                 "tef": 8640,
                 "tbs": 90,
                 "nt": ((64, 2), (64, 8), (96, 4), (96, 8), (104, 2), (128, 1), (128, 8)),
                 "nn": ((20, 20, 20, 20, 20, 20, 20)),
                 "csv_name": "all_" + csv_name,
                 "device": "gpu"
                 }
    for filename in filename_list:
        args_dict["filename"] = filename
        for ontn in ontn_list.keys():
            args_dict["ontn"] = ontn
            for oftn in ontn_list[ontn]:
                args_dict["oftn"] = oftn
                for i in range(5):
                    ok = False
                    for tcp in tcp_list:
                        args_dict["tcp"] = tcp
                        for tap in tap_list:
                            args_dict["tap"] = tap
                            ParamHolder().init_again(args_dict)
                            TimeHolder().init_again()
                            print(args_dict)
                            from src.envSim.offlineTaskDataProcess import offline_data_process
                            offline_data_process(ParamHolder().filename)
                            args_dict["cgr"] = ParamHolder().cpu_gpu_rate
                            from envSim.generateEnv import generate_cluster, generate_online_task_list, \
                                generate_offline_task_list
                            from scheduler.schedulerList import init_scheduler
                            cluster = generate_cluster()
                            online_task_list = generate_online_task_list()
                            offline_task_list = generate_offline_task_list()
                            while not torch.cuda.is_available():
                                pass
                            if not ok:
                                schedulers = []
                                schedulers.extend(init_scheduler(cluster, False))
                                schedulers.extend(init_scheduler(cluster, True))
                                ok = True
                            else:
                                schedulers = init_scheduler(cluster)
                            if not args_dict["test"]:
                                for scheduler in schedulers:
                                    p.apply_async(sim_run,
                                                  args=(scheduler, online_task_list, offline_task_list, deepcopy(args_dict),),
                                                  callback=callback)
                            else:
                                for scheduler in schedulers:
                                    print("单进程,测试用")
                                    sim_run(scheduler, online_task_list, offline_task_list)
    p.close()
    p.join()
    txt_to_csv("all_" + csv_name)


if __name__ == "__main__":
    run()
