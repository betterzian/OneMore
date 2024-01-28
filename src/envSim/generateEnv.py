from src.envSim.node import Node
from src.envSim.task import Task
import numpy as np
import pandas as pd
import random
import os
from src.envSim.timeSim import TimeHolder

# def generate_task_list(src_task = [], task_num = 500, src_task_file = str("../data_src/app_resources.csv")):
#     """
#     根据任务抽样来源队列，随机生成待调度队列
#     args：task_num 任务数量 , src_task_file = str("../data/src_task.csv")任务抽样文件地址
#     return： task_list 任务队列
#     保存task_list 至 ../data/task_list.csv
#     """
#     if len(src_task) == 0:
#         src_task = np.loadtxt(src_task_file, delimiter=',',dtype=float)
#     np.random.shuffle(src_task)
#     task_list = []
#     task_list_record = []
#     for i in range(task_num):
#         temp = np.array(src_task[random.randint(0, len(src_task) - 1)])
#         task_list.append(Task(i,temp))
# #         task_list_record.append(temp)
# #     task_list_record = np.array(task_list_record)
#     np.savetxt("../output/task_list.csv",task_list_record, delimiter=',')
#     return task_list


def generate_offline_task_list(src_task = [], task_num = 500, src_task_file = str("../data_src/offline_task/openb_pod_list_gpushare100.csv")):
    """
    根据任务抽样来源队列，随机生成待调度队列
    args：task_num 任务数量 , src_task_file = str("../data/src_task.csv")任务抽样文件地址
    return： task_list 任务队列
    保存task_list 至 ../data/task_list.csv
    """

    if task_num == 0:
        return []
    if len(src_task) == 0:
        src_task = pd.read_csv(src_task_file,header=0)
    # src_task = src_task["creation_time"]-src_task["creation_time"].max()
    src_task.drop("gpu_spec", axis=1, inplace=True)
    src_task.dropna(inplace=True)
    task_list = []
    task_list_record = []
    time_len = TimeHolder().get_time_left()
    for i in range(task_num):
        temp = src_task.iloc[random.randint(0, len(src_task) - 1)]
        task_list.append(Task(id=i,cpu=round(temp["cpu_milli"]/1000, 1),gpu=round(temp["num_gpu"]*temp["gpu_milli"]/1000, 1),time_len=int((temp["deletion_time"]-temp["scheduled_time"])/10 + 1),arrive_time=random.randint(0,time_len-1)))
    #     task_list_record.append(temp)
    # task_list_record = np.array(task_list_record)
    # np.savetxt("../output/offline_task_list.csv",task_list_record, delimiter=',')
    task_list = sorted(task_list,key=lambda task:-task.get_arrive_time())
    return task_list


def generate_online_task_list(task_num = 10):
    if task_num == 0:
        return []
    file_list = os.listdir('/disk7T/vis/code/OneMore/data_src/online_task/')
    length = len(file_list) - 1
    task_list = []
    task_list_record = []
    i = 0
    while i < task_num:
        file_name = file_list[random.randint(0,length)]
        temp = np.loadtxt('/disk7T/vis/code/OneMore/data_src/online_task/' + str(file_name),delimiter=",")
        if len(temp) != 8641:
            print(file_name,len(temp))
            continue
        task_list.append(Task(i,temp))
        task_list_record.append(temp)
        i += 1
    task_list_record = np.array(task_list_record)
    np.savetxt("../output/online_task_list.csv",task_list_record, delimiter=',')
    return task_list

def generate_src_task_list():
    tasks = pd.read_csv('/disk7T/vis/code/OneMore/data_src/container.csv',header=None)
    tasks = tasks.values
    return tasks

def generate_cluster(node_type = [(32, 4), (96, 8)], node_num = (10, 10)):
    """
    args：node_type =  [(32,4),(96,8)] node的资源种类
          node_num = (10,10) node不同的资源种类对应的数量
          time_block = 2 node的时间块数量
    return: node_list(第一位为gpu数量）
    保存node_list到 "../data/node_list.csv"
    """
    cluster = []
    count = 0
    for i in range(len(node_type)):
        for _ in range(node_num[i]):
            cluster.append(Node(count,node_type[i][0],node_type[i][1]))
            count += 1
    random.shuffle(cluster)
    for i in range(count):
        cluster[i].set_id(i)
    return cluster

if __name__ == "__main__":
    task_list = generate_online_task_list()
    task_list = generate_offline_task_list()
    cluster = generate_cluster()