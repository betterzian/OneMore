import sys
sys.path.append("../")
import os
import argparse
import time
from datetime import datetime
from src.envSim.TXTtoCSV import txt_to_csv

if __name__ == "__main__":
    current_time = datetime.now()
    parser = argparse.ArgumentParser(description='param')
    parser.add_argument('--num', type=int, default=10)
    args = parser.parse_args()
    csv_name = str(current_time.year) + "_" + str(current_time.month) + "_" + str(
        current_time.day) + "_" + str(current_time.hour) + "_" + str(current_time.minute) + "_" + str(
        current_time.second)
    count = 0
    file = open('../tmp/all_'+csv_name+'.txt', "w")
    file.close()
    for i in range(args.num):
        while i- count > 4:
            time.sleep(60)
            count = 0
            with open('../tmp/all_'+csv_name+'.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    count += 1
    while args.num > count:
        time.sleep(60)
        count = 0
        with open('../tmp/all_' + csv_name + '.txt', 'r', encoding='utf-8') as f:
            for line in f:
                count += 1
    txt_to_csv("all_"+csv_name)

