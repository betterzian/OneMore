import sys
sys.path.append("../")
import argparse
from datetime import datetime
from envSim.offlineTaskDataProcess import offline_data_process


def generate_sim_param(args):
    current_time = datetime.now()
    cgr = offline_data_process(args.filename)
    with open("simParam.py", "w") as f:
        f.write(f"__zero__ = {0.000001}\n")
        f.write(f"__time_len__ = {args.tl}\n")
        f.write(f"__time_init_flag__ = {args.tif}\n")
        f.write(f"__time_end_flag__ = {args.tef}\n")
        f.write(f"__time_can_predict__ = {args.tcp}\n")
        f.write(f"__time_block_size__ = {args.tbs}\n")
        f.write(f"__time_accurately_predict__ = {args.tap}\n")
        f.write(f"__online_task_num__ = {args.ontn}\n")
        f.write(f"__offline_task_num__ = {args.oftn}\n")
        f.write(f"__cpu_gpu_rate__ = {cgr}\n")
        f.write(f"__node_type__ = {args.nt}\n")
        f.write(f"__node_num__ = {args.nn}\n")
        f.write(f"args = {vars(args)}\n")
        f.write(f"__filename__ = '{args.filename}'\n")
        f.write(f'__csv_name__= "' + f'{str(current_time.year)+"_"+str(current_time.month)+"_"+str(current_time.day)+"_"+str(current_time.hour)+"_"+str(current_time.minute)+"_"+str(current_time.second)}"\n')
    import src.simParam

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='param')
    parser.add_argument('--mbool', type=int, default=True)
    parser.add_argument('--tl', type=int, default=17280)
    parser.add_argument('--tif', type=int, default=0)
    parser.add_argument('--tef', type=int, default=8640)
    parser.add_argument('--tcp', type=int, default=8640)
    parser.add_argument('--tbs', type=int, default=90)
    parser.add_argument('--tap', type=int, default=90)
    parser.add_argument('--ontn', type=int, default=5)
    parser.add_argument('--oftn', type=int, default=800)
    parser.add_argument('--filename', type=str, default="openb_pod_list_multigpu50")
    parser.add_argument('--nt', type=list, default=((64, 2),(64, 8),(96, 4),(96, 8),(104, 2),(128, 1),(128, 8)))
    parser.add_argument('--nn', type=tuple, default=(10,10,10,10,10,10,10))
    args = parser.parse_args()
    generate_sim_param(args)
    from simRun import sim_run
    sim_run(args.mbool)





