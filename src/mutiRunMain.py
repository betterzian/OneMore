import sys
sys.path.append("../")
import os
import time
import argparse
from datetime import datetime
from src.envSim.TXTtoCSV import txt_to_csv


if __name__ == "__main__":
    current_time = datetime.now()
    parser = argparse.ArgumentParser(description='param')
    parser.add_argument('--num', type=int, default=60)
    args = parser.parse_args()
    csv_name = str(current_time.year) + "_" + str(current_time.month) + "_" + str(
        current_time.day) + "_" + str(current_time.hour) + "_" + str(current_time.minute) + "_" + str(
        current_time.second)
    count = 0
    file = open('../tmp/all_'+csv_name+'.txt', "w")
    file.close()
    tcp_list = [1080,2160,4320,8640]
    tap_list = [30,90,180,360]
    ontn_list  = {1000:[600,700,800],3000:[600,700,800],5000:[600,700,800]}
    filename_list = ["node", "openb_pod_list_gpushare100", "openb_pod_list_multigpu50"]
    sum_process = args.num
    now_process = 0
    for filename in filename_list:
        for ontn in ontn_list.keys():
            for oftn in ontn_list[ontn]:
                for tcp in tcp_list:
                    for tap in tap_list:
                        os.system(f"python main.py --filename={filename} --tap={tap} --tcp={tcp} --oftn={oftn} --ontn={ontn} -- --csv_name=all_"+csv_name+" &")
                        now_process += 14
                        while now_process - count > sum_process:
                            time.sleep(60)
                            count = 0
                            with open('../tmp/all_'+csv_name+'.txt', 'r', encoding='utf-8') as f:
                                for line in f:
                                    count += 1
    txt_to_csv("all_"+csv_name)
