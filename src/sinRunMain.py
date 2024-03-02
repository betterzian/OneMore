import sys
sys.path.append("../")

if __name__ == '__main__':
    import argparse
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
    parser.add_argument('--oftn', type=int, default=800)
    parser.add_argument('--filename', type=str, default="node")
    parser.add_argument('--nt', type=list, default=((64, 2),(64, 8),(96, 4),(96, 8),(104, 2),(128, 1),(128, 8)))
    parser.add_argument('--nn', type=tuple, default=(10,10,10,10,10,10,10))
    parser.add_argument('--csv_name', type=str, default=None)
    args = parser.parse_args()
    args_dict = vars(args)
    from simRun import sim_run
    sim_run(args_dict)











