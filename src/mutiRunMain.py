import sys

sys.path.append("../")
from datetime import datetime
import torch.multiprocessing as mp
from src.envSim.TXTtoCSV import txt_to_csv
from src.simRun import sim_run


def run():
    from src.envSim.simParam import ParamHolder
    ParamHolder()
    from src.envSim.timeSim import TimeHolder
    TimeHolder(wait=True)
    current_time = datetime.now()
    csv_name = str(current_time.year) + "_" + str(current_time.month) + "_" + str(
        current_time.day) + "_" + str(current_time.hour) + "_" + str(current_time.minute) + "_" + str(
        current_time.second)
    test = True
    tcp_list = [1080, 2160, 4320, 8640]
    tap_list = [30, 90, 180, 360]
    ontn_list = {1000: [600, 700, 800], 3000: [600, 700, 800], 5000: [600, 700, 800]}
    filename_list = ["openb_pod_list_multigpu50", "node", "openb_pod_list_gpushare100", ]
    p = mp.Pool(70)
    args_dict = {"test": False,
                 "tl": 17280,
                 "tif": 0,
                 "tef": 8640,
                 "tbs": 90,
                 "nt": ((64, 2), (64, 8), (96, 4), (96, 8), (104, 2), (128, 1), (128, 8)),
                 "nn": ((20, 20, 20, 20, 20, 20, 20)),
                 "csv_name": "all_" + csv_name,
                 "device": "cpu"
                 }
    for filename in filename_list:
        args_dict["filename"] = filename
        for tcp in tcp_list:
            args_dict["tcp"] = tcp
            for tap in tap_list:
                args_dict["tap"] = tap
                for ontn in ontn_list.keys():
                    args_dict["ontn"] = ontn
                    for oftn in ontn_list[ontn]:
                        args_dict["oftn"] = oftn
                        for i in range(5):
                            ParamHolder().init_again(args_dict)
                            TimeHolder().init_again()
                            from src.envSim.offlineTaskDataProcess import offline_data_process
                            offline_data_process(ParamHolder().filename)
                            args_dict["cgr"] = ParamHolder().cpu_gpu_rate
                            from envSim.generateEnv import generate_cluster, generate_online_task_list, \
                                generate_offline_task_list
                            from scheduler.schedulerList import init_scheduler
                            cluster = generate_cluster()
                            online_task_list = generate_online_task_list()
                            offline_task_list = generate_offline_task_list()
                            schedulers = init_scheduler(cluster)
                            if not test:
                                for scheduler in schedulers:
                                    p.apply_async(sim_run, args=(scheduler, online_task_list, offline_task_list,))
                            else:
                                for scheduler in schedulers:
                                    print("单进程,测试用")
                                    sim_run(scheduler, online_task_list, offline_task_list)
    p.close()
    p.join()
    txt_to_csv("all_" + csv_name)


if __name__ == "__main__":
    run()
