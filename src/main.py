import sys
sys.path.append("../")
import argparse
from src.envSim.simParam import ParamHolder
from src.envSim.offlineTaskDataProcess import offline_data_process

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='param')
    parser.add_argument('--test', action='store_false')
    parser.add_argument('--gather', action='store_false')
    parser.add_argument('--tl', type=int, default=17280)
    parser.add_argument('--tif', type=int, default=0)
    parser.add_argument('--tef', type=int, default=8640)
    parser.add_argument('--tcp', type=int, default=8640)
    parser.add_argument('--tbs', type=int, default=90)
    parser.add_argument('--tap', type=int, default=90)
    parser.add_argument('--ontn', type=int, default=1)
    parser.add_argument('--oftn', type=int, default=1500)
    parser.add_argument('--filename', type=str, default="openb_pod_list_multigpu50")
    parser.add_argument('--nt', type=list, default=((64, 2),(64, 8),(96, 4),(96, 8),(104, 2),(128, 1),(128, 8)))
    parser.add_argument('--nn', type=tuple, default=(20,20,20,20,20,20,20))
    parser.add_argument('--csv_name', type=str, default=None)
    args = parser.parse_args()
    ParamHolder(args)
    offline_data_process(ParamHolder().filename)
    # if args.gather:
    #     pass
    # else:
    from simRun import sim_run
    sim_run(False)
    #sim_run(args.test)
    with open('../tmp/'+ParamHolder().csv_name+'.txt', 'a') as file:
        file.write('ok\n')







