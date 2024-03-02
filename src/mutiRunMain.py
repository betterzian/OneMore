import sys
sys.path.append("../")
from multiprocessing import Pool
from simRun import sim_run
from datetime import datetime
from src.envSim.TXTtoCSV import txt_to_csv

if __name__ == "__main__":
    current_time = datetime.now()
    csv_name = str("all_")+str(current_time.year) + "_" + str(current_time.month) + "_" + str(
        current_time.day) + "_" + str(current_time.hour) + "_" + str(current_time.minute) + "_" + str(
        current_time.second)
    args_list = {
        "test":True,
        "tl":17280,
        "tif":0,
        "tef":8640,
        "tcp":8640,
        "tbs":90,
        "tap":90,
        "ontn":5000,
        "oftn":700,
        "filename":"openb_pod_list_multigpu50",
        "nt":((64, 2),(64, 8),(96, 4),(96, 8),(104, 2),(128, 1),(128, 8)),
        "nn":(10,10,10,10,10,10,10),
        "csv_name":csv_name,
    }
    p = Pool(60)
    oftn = [600,650,700,750,800]
    for i in oftn:
        args_list["oftn"] = i
        p.apply_async(sim_run, args=args_list)
    p.close()
    p.join()
    txt_to_csv(args_list["csv_name"])

