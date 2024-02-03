import argparse
import sys
sys.path.append("../")

def generate_sim_param(args):
    with open("simParam.py", "w") as f:
        f.write(f"__zero__ = {0.000001}\n")
        f.write(f"__time_len__ = {args.tl}\n")
        f.write(f"__time_init_flag__ = {args.tif}\n")
        f.write(f"__time_can_predict__ = {args.tcp}\n")
        f.write(f"__time_block_size__ = {args.tbs}\n")
        f.write(f"__time_accurately_predict__ = {args.tap}\n")
        f.write(f"__online_task_num__ = {args.ontn}\n")
        f.write(f"__offline_task_num__ = {args.oftn}\n")
        f.write(f"__cpu_gpu_rate__ = {args.cgr}\n")
        f.write(f"__node_type__ = {args.nt}\n")
        f.write(f"__node_num__ = {args.nn}\n")
        f.write(f"args = {vars(args)}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='param')
    parser.add_argument('--tl', type=int, default=8640)
    parser.add_argument('--tif', type=int, default=0)
    parser.add_argument('--tcp', type=int, default=3600)
    parser.add_argument('--tbs', type=int, default=90)
    parser.add_argument('--tap', type=int, default=90)
    parser.add_argument('--ontn', type=int, default=1000)
    parser.add_argument('--oftn', type=int, default=1500)
    parser.add_argument('--cgr', type=int, default=4)
    parser.add_argument('--nt', type=list, default=[(24, 4), (48, 8)])
    parser.add_argument('--nn', type=tuple, default=[72, 72])
    args = parser.parse_args()
    generate_sim_param(args)
    from simRun import sim_run
    sim_run(False)





