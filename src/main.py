import sys
sys.path.append("../")
import argparse
from src.envSim.simParam import ParamHolder
from src.envSim.offlineTaskDataProcess import offline_data_process
import json
import torch

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description='param')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--gather', action='store_false')
    parser.add_argument('--tl', type=int, default=17280)
    parser.add_argument('--tif', type=int, default=0)
    parser.add_argument('--tef', type=int, default=8640)
    parser.add_argument('--tcp', type=int, default=8640)
    parser.add_argument('--tbs', type=int, default=90)
    parser.add_argument('--tap', type=int, default=90)
    parser.add_argument('--ontn', type=int, default=5000)
    parser.add_argument('--oftn', type=int, default=700)
    parser.add_argument('--filename', type=str, default="openb_pod_list_multigpu50")
    parser.add_argument('--nt', type=list, default=((64, 2),(64, 8),(96, 4),(96, 8),(104, 2),(128, 1),(128, 8)))
    parser.add_argument('--nn', type=tuple, default=(10,10,10,10,10,10,10))
    parser.add_argument('--csv_name', type=str, default=None)
    args = parser.parse_args()
    args_dict = vars(args)
    ParamHolder(args_dict)
    ParamHolder().cpu_gpu_rate = offline_data_process(ParamHolder().filename)
    args_dict["cgr"] = ParamHolder().cpu_gpu_rate
    args_dict["csv_name"] = ParamHolder().csv_name
    args_dict["reader"] = 0
    # ok = False
    # while not ok:
    #     if os.path.exists("../tmp/param/args.json"):
    #         with open("../tmp/param/args.json", "r+") as json_file:
    #             temp_args_list = json.load(json_file)
    #             if temp_args_list["reader"] == 2:
    #                 json_file.seek(0)
    #                 json_file.truncate()
    #                 json.dump(args_dict, json_file)
    #                 ok = True
    #         time.sleep(10)
    #     else:
    with open("../tmp/param/args.json", "w") as json_file:
        json.dump(args_dict, json_file)
        ok = True
    # if args.gather:
    #     pass
    # else:
    from simRun import sim_run
    sim_run(args.test)
    with open('../tmp/'+ParamHolder().csv_name+'.txt', 'a') as file:
        file.write('ok\n')







