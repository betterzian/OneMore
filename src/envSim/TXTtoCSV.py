import pandas as pd
import argparse
import os
import csv


def txt_to_csv(csv_name):
    file = pd.read_csv("../output/scheduler_result_" + csv_name + ".txt", header=0, sep='\t',
                       encoding="utf-8", quoting=csv.QUOTE_NONE, escapechar=',')
    file = file[file['cpu_rate'] != "cpu_rate"]
    file.loc[file['can_predict'] == 'False', 'time_can_predict'] = 0
    file.loc[file['can_predict'] == 'False', 'time_accurately_predict'] = 0
    file.to_csv("../output/scheduler_result_" + csv_name + ".csv", index=False, header=True, sep=',')
    os.remove("../output/scheduler_result_" + csv_name + ".txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='param')
    parser.add_argument('--csv_name', type=str, default="test")
    args = parser.parse_args()
    txt_to_csv(args.csv_name)