import numpy as np
import pandas as pd
import os
import json
from src.envSim.simParam import ParamHolder
from src.scheduler.myAlgorithm.generateValue.generateDiscreteValue import generate_state

def float_to_int(x):
    return int(x*10)

def offline_data_process(filename):
    if os.path.exists("../srcData/state_value/"+filename+"/state_int.csv"):
        with open("../srcData/state_value/"+filename+"/data.json", "r") as json_file:
            data = json.load(json_file)
            ParamHolder().cpu_gpu_rate = data["cgr"]
            return
    elif not os.path.exists("../srcData/state_value/"+filename):
        os.mkdir("../srcData/state_value/"+filename)
    src = pd.read_csv("../srcData/offline_task/"+filename+".csv", header=0)
    src['gpu_milli'] = src['num_gpu'] * src['gpu_milli']
    task_prob_int = np.zeros((int(src["cpu_milli"].max() / 1000)+1,int(src["gpu_milli"].max() / 1000)+1))
    task_prob_float = np.zeros((int(src["cpu_milli"].max() / 1000)+1,10))
    off_task_list = []
    src = src[src["gpu_milli"] > ParamHolder().zero]
    for temp in src.iloc:
        if (temp["gpu_milli"] / 1000) < 1:
            task_prob_float[int(temp["cpu_milli"] / 1000)][float_to_int(round(temp["gpu_milli"] / 1000, 1))] += 1
        else:
            task_prob_int[int(temp["cpu_milli"] / 1000)][int(temp["gpu_milli"] / 1000)] += 1
        off_task_list.append([int(temp["cpu_milli"] / 1000), round(temp["gpu_milli"] / 1000, 1)])
    off_task_list = np.array(off_task_list)
    temp = np.sum(off_task_list,axis=0)
    np.savetxt("../srcData/offline_task/"+filename+"off_task_list.csv", off_task_list, delimiter=',')
    np.savetxt("../srcData/state_value/"+filename+"/task_prob_int.csv", task_prob_int, delimiter=',')
    np.savetxt("../srcData/state_value/"+filename+"/task_prob_float.csv", task_prob_float, delimiter=',')
    generate_state(filename)
    data = {
        "cgr": int(temp[0] / temp[1])
    }
    with open("../srcData/state_value/"+filename+"/data.json", "w") as json_file:
        json.dump(data, json_file)
    ParamHolder().cpu_gpu_rate = data["cgr"]
    return

if __name__ == "__main__":
    src = np.loadtxt("../srcData/offline_task/off_task_test.csv", delimiter=",")
    temp = src.max(axis=0)
    task_prob_int = np.zeros((int(temp[0]+1),int(temp[1]+1)))
    task_prob_float = np.zeros((int(temp[0]+1),10))
    for temp in src:
        if temp[1] < 1:
            task_prob_float[int(temp[0])][float_to_int(temp[1])] = temp[2]
        else:
            task_prob_int[int(temp[0])][int(temp[1])] = temp[2]
    off_task_list = []
    for temp in src:
        for i in range(int(temp[2])):
            off_task_list.append([temp[0],temp[1]])
    off_task_list = np.array(off_task_list)
    np.savetxt("../srcData/offline_task/off_task_list.csv", off_task_list, delimiter=',')
    np.savetxt("../srcData/state_value/task_prob_int.csv", task_prob_int, delimiter=',')
    np.savetxt("../srcData/state_value/task_prob_float.csv", task_prob_float, delimiter=',')
