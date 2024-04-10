import math
import os
import json
import numpy as np
import pandas as pd
from src.envSim.simParam import ParamHolder
from src.scheduler.myAlgorithm.generateValue.generateDiscreteValue import generate_state


def float_to_int(x):
    return int(x * 10)


def send_data_to_param(data):
    ParamHolder().cpu_gpu_rate = data["cgr"]
    ParamHolder().avgsize = data["avgsize"]
    ParamHolder().zero_task = data["zero_task"]


def offline_data_process(filename,only_cgr=False):
    ParamHolder()
    if os.path.exists("../srcData/state_value/" + filename + "/data.json"):
        with open("../srcData/state_value/" + filename + "/data.json", "r") as json_file:
            data = json.load(json_file)
            send_data_to_param(data)
            if filename[:5] != "param":
                return data["cgr"]
            elif os.path.exists(f"../srcData/state_value/{filename}/state{ParamHolder().prob}.csv"):
                return data["cgr"]
    elif not os.path.exists("../srcData/state_value/" + filename):
        os.mkdir("../srcData/state_value/" + filename)
    if filename[:5] == "param":
        off_task_list = pd.read_csv("../srcData/state_value/" + filename + "/off_task_list.csv", header=0)
        off_task_list = off_task_list.values
        temp = np.average(off_task_list, axis=0)
    else:
        src = pd.read_csv("../srcData/offline_task/" + filename + ".csv", header=0)
        src['gpu_milli'] = src['num_gpu'] * src['gpu_milli']
        task_prob_int = np.zeros((int(src["cpu_milli"].max() / 1000) + 1, int(src["gpu_milli"].max() / 1000) + 2))
        task_prob_float = np.zeros((int(src["cpu_milli"].max() / 1000) + 1, 10))
        off_task_list = []
        src = src[src["gpu_milli"] > ParamHolder().zero]
        for temp in src.iloc:
            if (temp["gpu_milli"] / 1000) < 1:
                task_prob_float[int(temp["cpu_milli"] / 1000)][float_to_int(round(temp["gpu_milli"] / 1000, 1))] += 1
            else:
                task_prob_int[int(temp["cpu_milli"] / 1000)][int(temp["gpu_milli"] / 1000)] += 1
            off_task_list.append([round(temp["cpu_milli"] / 1000), round(temp["gpu_milli"] / 1000, 1)])
        off_task_list = np.array(off_task_list)
        np.savetxt("../srcData/state_value/" + filename + "/off_task_list.csv", off_task_list, delimiter=',')
        np.savetxt("../srcData/state_value/" + filename + "/task_prob_int.csv", task_prob_int, delimiter=',')
        np.savetxt("../srcData/state_value/" + filename + "/task_prob_float.csv", task_prob_float, delimiter=',')
        off_task_list_copy = off_task_list[off_task_list[:, 0] < 64]
        temp = np.average(off_task_list, axis=0)
        temp_copy = np.average(off_task_list_copy, axis=0)

    data = {
        "cgr": max(1, int(temp[0] / temp[1])),
        "avgsize": math.ceil(temp[0] + max(1, int(temp[0] / temp[1])) * temp[1]),
    }
    if filename[:5] != "param":
        data["zero_task"] = int(min(ParamHolder().all_node_cpu / temp_copy[0], ParamHolder().all_node_gpu / temp_copy[1]) / 110 * 100)
    if not only_cgr:
        send_data_to_param(data)
    if filename[:5] == "param":
        generate_state()
    else:
        cal_smaller_task_count(filename)
    if not only_cgr:
        with open("../srcData/state_value/" + filename + "/data.json", "w") as json_file:
            json.dump(data, json_file)
    return data["cgr"]


def cal_smaller_task_count(filename=None,task_prob_float=None, task_prob_int=None):
    """
    计算当前任务概率场景下小于等于当前任务大小的个数，为求概率函数做准备。
    args：task_prob_file = str("../data/task_prob.csv")
    return: smaller_task_count
    保存smaller_task_count 至 ../data/smaller_task_count.csv
    """

    if task_prob_float is None:
        task_prob_float_file = str("../srcData/state_value/" + filename + "/task_prob_float.csv")
        task_prob_float = np.loadtxt(task_prob_float_file, delimiter=',')
    if task_prob_int is None:
        task_prob_int_file = str("../srcData/state_value/" + filename + "/task_prob_int.csv")
        task_prob_int = np.loadtxt(task_prob_int_file, delimiter=',')
    smaller_task_count_float = np.zeros((200, 10))
    smaller_task_count_float[0][0] = task_prob_float[0][0]
    for i in range(1, len(smaller_task_count_float)):
        if i >= len(task_prob_float):
            temp = 0
        else:
            temp = task_prob_float[i][0]
        smaller_task_count_float[i][0] = smaller_task_count_float[i - 1][0] + temp
    for j in range(1, len(smaller_task_count_float[0])):
        if j >= len(task_prob_float[0]):
            temp = 0
        else:
            temp = task_prob_float[0][j]
        smaller_task_count_float[0][j] = smaller_task_count_float[0][j - 1] + temp
    for i in range(1, len(smaller_task_count_float)):
        for j in range(1, len(smaller_task_count_float[0])):
            if j >= len(task_prob_float[0]) or i >= len(task_prob_float):
                temp = 0
            else:
                temp = task_prob_float[i][j]
            smaller_task_count_float[i][j] = smaller_task_count_float[i][j - 1] + smaller_task_count_float[i - 1][j] - \
                                             smaller_task_count_float[i - 1][j - 1] + temp
    smaller_task_count_int = np.zeros((200, 20))
    smaller_task_count_int[0][0] = task_prob_int[0][0]
    smaller_task_count_int[0][1] = smaller_task_count_float[0][9] + task_prob_int[0][1] + smaller_task_count_int[0][0]
    for i in range(1, len(smaller_task_count_int)):
        if i >= len(task_prob_int):
            temp = 0
        else:
            temp = task_prob_int[i][0]
        smaller_task_count_int[i][0] = smaller_task_count_int[i - 1][0] + temp
        if i >= len(task_prob_int):
            temp = 0
        else:
            temp = task_prob_int[i][1]
        smaller_task_count_int[i][1] = smaller_task_count_int[i - 1][1] + temp + \
                                       smaller_task_count_float[i][9] - smaller_task_count_float[i - 1][9]
    for j in range(2, len(smaller_task_count_int[0])):
        if j >= len(task_prob_int[0]):
            temp = 0
        else:
            temp = task_prob_int[0][j]
        smaller_task_count_int[0][j] = smaller_task_count_int[0][j - 1] + temp
    for i in range(1, len(smaller_task_count_int)):
        for j in range(2, len(smaller_task_count_int[0])):
            if j >= len(task_prob_int[0]) or i >= len(task_prob_int):
                temp = 0
            else:
                temp = task_prob_int[i][j]
            smaller_task_count_int[i][j] = smaller_task_count_int[i][j - 1] + smaller_task_count_int[i - 1][j] - \
                                           smaller_task_count_int[i - 1][j - 1] + temp

    np.savetxt("../srcData/state_value/" + filename + "/smaller_task_count_int.csv", smaller_task_count_int,
               delimiter=',')
    np.savetxt("../srcData/state_value/" + filename + "/smaller_task_count_float.csv", smaller_task_count_float,
               delimiter=',')

    smaller_task_count = np.zeros((200, 200))
    for i in range(len(smaller_task_count_int)):
        for j in range(len(smaller_task_count_int[0])):
            smaller_task_count[i][j * 10] = smaller_task_count_int[i][j]
    for i in range(len(smaller_task_count_float)):
        for j in range(len(smaller_task_count_float[0])):
            smaller_task_count[i][j] = smaller_task_count_float[i][j]
    smaller_task_count = np.round(smaller_task_count * 1.0 / smaller_task_count.max(),3)
    np.savetxt("../srcData/state_value/" + filename + "/smaller_task_count.csv", smaller_task_count,
               delimiter=',')
    return


if __name__ == "__main__":
    offline_data_process("openb_pod_list_multigpu50")
