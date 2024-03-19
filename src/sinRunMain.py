import sys
sys.path.append("../")
from src.simRun import sim_run

def run(args_dict):
    from src.envSim.simParam import ParamHolder
    ParamHolder(args_dict)
    file = open('../tmp/all_'+ParamHolder().csv_name+'.txt', "w")
    file.close()
    from src.envSim.offlineTaskDataProcess import offline_data_process
    offline_data_process(ParamHolder().filename)
    args_dict["cgr"] = ParamHolder().cpu_gpu_rate
    args_dict["csv_name"] = ParamHolder().csv_name
    from envSim.generateEnv import generate_cluster, generate_online_task_list, generate_offline_task_list
    from scheduler.schedulerList import init_scheduler
    from multiprocessing import Pool
    from src.envSim.TXTtoCSV import txt_to_csv
    cluster = generate_cluster()
    online_task_list = generate_online_task_list()
    offline_task_list = generate_offline_task_list()
    schedulers = init_scheduler(cluster)
    if not args_dict["test"]:
        p = Pool(len(schedulers))
        for scheduler in schedulers:
            p.apply_async(sim_run, args=(scheduler, online_task_list, offline_task_list,))
        p.close()
        p.join()
    else:
        for scheduler in schedulers:
            print("单进程,测试用")
            sim_run(scheduler, online_task_list, offline_task_list)
    if ParamHolder().csv_name[:3] != "all":
        txt_to_csv(ParamHolder().csv_name)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='param')
    parser.add_argument('--test', action='store_false')
    parser.add_argument('--gather', action='store_false')
    parser.add_argument('--tl', type=int, default=17280)
    parser.add_argument('--tif', type=int, default=0)
    parser.add_argument('--tef', type=int, default=8640)
    parser.add_argument('--tcp', type=int, default=8640)
    parser.add_argument('--tbs', type=int, default=90)
    parser.add_argument('--tap', type=int, default=90)
    parser.add_argument('--ontn', type=int, default=10)
    parser.add_argument('--oftn', type=int, default=2000)
    parser.add_argument('--filename', type=str, default="openb_pod_list_gpushare100")
    # parser.add_argument('--nt', type=list, default=((42,4),))
    # parser.add_argument('--nn', type=tuple, default=((70,)))
    parser.add_argument('--nt', type=list, default=((64, 2),(64, 8),(96, 4),(96, 8),(104, 2),(128, 1),(128, 8)))
    parser.add_argument('--nn', type=tuple, default=((20,20,20,20,20,20,20)))
    parser.add_argument('--csv_name', type=str, default=None)
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict["device"] = "cpu"
    run(args_dict)











