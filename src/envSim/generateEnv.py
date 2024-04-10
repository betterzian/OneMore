import os
import random
import numpy as np
from src.envSim.node import Node
from src.envSim.task import Task
from src.envSim.timeSim import TimeHolder
from src.envSim.simParam import ParamHolder

def generate_task_prob(min=0, max=10, size=(10 , 10), small_task = (0, 0)):
    """
    生成 任务出现的概率
    args：min任务最小出现次数, max任务最大出现次数, size=(10 , 10)任务大小范围矩阵,small_task = (0,0)不会出现的小任务的大小范围矩阵
    return：task_prob任务概率, src_task任务抽样来源队列
    """
    task_prob = np.random.randint(min, max, size)
    for i in range(1, len(task_prob)):
        task_prob[i][i] = 0
        task_prob[i - 1][i] = 0
        task_prob[i][i - 1] = 0
    for i in range(int((size[0] -1) * (size[1] - 1) / 4)):
        task_prob[random.randint(0, size[0] -1)][random.randint(0, size[1] - 1)] = 0
    task_prob[0][0] = 0
    for i in range(small_task[0]):
        for j in range(small_task[1]):
            task_prob[i][j] = 0
    src_task = []
    for i in range(len(task_prob)):
        for j in range(len(task_prob[i])):
            for k in range(task_prob[i][j]):
                src_task.append([i, j])
    src_task = np.array(src_task)
    if not os.path.exists("../srcData/state_value/" + ParamHolder().filename):
        os.mkdir("../srcData/state_value/" + ParamHolder().filename)
    np.savetxt("../srcData/state_value/" + ParamHolder().filename + "/off_task_list.csv", src_task, delimiter=',')
    np.savetxt("../srcData/state_value/" + ParamHolder().filename + "/task_prob.csv", task_prob, delimiter=',')

def generate_offline_task_list(temp_list=[], task_num=None, src_task_file=None, all_bool=False):
    if task_num == 0:
        return []
    if task_num is None:
        task_num = ParamHolder().offline_task_num
    if src_task_file is None:
        src_task_file = f"../srcData/state_value/{ParamHolder().filename}/off_task_list.csv"
    time_len = TimeHolder().get_fake_time_left()
    if len(temp_list) == 0:
        src_task = np.loadtxt(src_task_file, delimiter=',', dtype=float)
        if all_bool:
            return src_task
        temp_list = np.random.choice(range(len(src_task)), size=task_num, replace=True)
        temp_list = src_task[temp_list]
    task_list = []
    for i in range(len(temp_list)):
        # if i < 1100:
        if i < ParamHolder().zero_task:
            arrive_time = 0
        elif ParamHolder().filename[:5] == "param":
            arrive_time = 0
        else:
            arrive_time = random.randint(1, time_len - 1)
        if ParamHolder().filename[:5] == "param":
            temp_time_len = 1
        else:
            temp_time_len = random.randint(90, time_len - 1)
        temp = temp_list[i]
        task_list.append(
            Task(id=i, cpu=temp[0], gpu=temp[1],
                 time_len=temp_time_len,
                 arrive_time=arrive_time)
        )
    task_list = sorted(task_list, key=lambda task: -task.get_arrive_time())
    return task_list


def generate_online_task_list(task_num=None):
    if task_num is None:
        task_num = ParamHolder().online_task_num
    if task_num == 0:
        return []
    file_list = os.listdir('../srcData/online_task/')
    length = len(file_list) - 1
    task_list = []
    task_list_record = []
    i = 0
    while i < task_num:
        file_name = file_list[random.randint(0, length)]
        temp = np.loadtxt('../srcData/online_task/' + str(file_name), delimiter=",")
        if len(temp) != 17281:
            continue
        task_list.append(Task(i, temp))
        task_list_record.append(temp)
        i += 1
    return task_list


def generate_cluster(node_type=None, node_num=None):
    if node_type is None:
        node_type = ParamHolder().node_type
    if node_num is None:
        node_num = ParamHolder().node_num
    cluster = []
    count = 0
    for i in range(len(node_type)):
        for _ in range(node_num[i]):
            cluster.append(Node(count, node_type[i][0], node_type[i][1]))
            count += 1
    random.shuffle(cluster)
    for i in range(count):
        cluster[i].set_id(i)
    return cluster


if __name__ == "__main__":
    online_task_list = generate_online_task_list(task_num=10)
    offline_task_list = generate_offline_task_list(task_num=10)
    cluster = generate_cluster(node_type=[(32, 4), (96, 8)], node_num=(10, 10))
