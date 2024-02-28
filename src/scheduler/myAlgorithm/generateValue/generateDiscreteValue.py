import numpy as np
from src.envSim.simParam import ParamHolder
from tqdm import tqdm

def float_to_int(x):
    return int(x*10)


def cal_smaller_task_count(task_prob_float = [], task_prob_int = [], filename = None):
    """
    计算当前任务概率场景下小于等于当前任务大小的个数，为求概率函数做准备。
    args：task_prob_file = str("../data/task_prob.csv")
    return: smaller_task_count
    保存smaller_task_count 至 ../data/smaller_task_count.csv
    """

    if len(task_prob_float) == 0:
        task_prob_float_file = str("../srcData/state_value/"+filename+"/task_prob_float.csv")
        task_prob_float = np.loadtxt(task_prob_float_file, delimiter=',')
    if len(task_prob_int) == 0:
        task_prob_int_file = str("../srcData/state_value/"+filename+"/task_prob_int.csv")
        task_prob_int = np.loadtxt(task_prob_int_file, delimiter=',')
    smaller_task_count_float = np.zeros(task_prob_float.shape)
    smaller_task_count_float[0][0] = task_prob_float[0][0]
    for i in range(1, len(smaller_task_count_float)):
        smaller_task_count_float[i][0] = smaller_task_count_float[i - 1][0] + task_prob_float[i][0]
    for j in range(1,len(smaller_task_count_float[0])):
        smaller_task_count_float[0][j] = smaller_task_count_float[0][j - 1] + task_prob_float[0][j]
    for i in range(1, len(smaller_task_count_float)):
        for j in range(1, len(smaller_task_count_float[0])):
            smaller_task_count_float[i][j] = smaller_task_count_float[i][j - 1] + smaller_task_count_float[i - 1][j] - smaller_task_count_float[i - 1][j - 1] + task_prob_float[i][j]
    smaller_task_count_int = np.zeros(task_prob_int.shape)
    smaller_task_count_int[0][0] = task_prob_int[0][0]
    smaller_task_count_int[0][1] = smaller_task_count_float[0][9] + task_prob_int[0][1] + smaller_task_count_int[0][0]
    for i in range(1,len(smaller_task_count_int)):
        smaller_task_count_int[i][0] = smaller_task_count_int[i - 1][0] + task_prob_int[i][0]
        smaller_task_count_int[i][1] = smaller_task_count_int[i - 1][1] + task_prob_int[i][1] + smaller_task_count_float[i][9] - smaller_task_count_float[i-1][9]
    for j in range(2, len(smaller_task_count_int[0])):
        smaller_task_count_int[0][j] = smaller_task_count_int[0][j - 1] + task_prob_int[0][j]
    for i in range(1, len(smaller_task_count_int)):
        for j in range(2, len(smaller_task_count_int[0])):
            smaller_task_count_int[i][j] = smaller_task_count_int[i][j - 1] + smaller_task_count_int[i - 1][j] - \
                                             smaller_task_count_int[i - 1][j - 1] + task_prob_int[i][j]

    np.savetxt("../srcData/state_value/"+filename+"/smaller_task_count_int.csv", smaller_task_count_int, delimiter=',')
    np.savetxt("../srcData/state_value/"+filename+"/smaller_task_count_float.csv", smaller_task_count_float, delimiter=',')
    return smaller_task_count_int.max()

def beta_F(smaller_task_count, input, num):
    if input.min() < 0:
        return 0
    else:
        if input[0] > len(smaller_task_count) - 1:
            input[0] = len(smaller_task_count) - 1
        if input[1] > len(smaller_task_count[0]) - 1:
            input[1] = len(smaller_task_count[0]) - 1
        if num == 0:
            return 0
        return smaller_task_count[input[0]][input[1]] * 1.0 / num

def beta_f_F(task_prob, smaller_task_count, input, top):
    if input.min() < 0 or input[0] > (len(task_prob) - 1) or input[1] > (len(task_prob[0]) - 1):
        return 0
    else:
        if smaller_task_count[min(top[0], len(smaller_task_count) - 1)][min(top[1], len(smaller_task_count[0]) - 1)] == 0:
            return 0
        return task_prob[input[0]][input[1]] * 1.0 / smaller_task_count[min(top[0], len(smaller_task_count) - 1)][min(top[1], len(smaller_task_count[0]) - 1)]

def state_value_float(sum, task_prob_float = [], smaller_task_count_float = [], prob = 0.05, state_size = (97, 10), filename = None):
    """
    计算二维状态下的状态价值
    args： state_size = (96,96) 状态空间的大小
          task_prob_file = str("../data/task_prob.csv")
          smaller_task_count_file = str("../data/smaller_task_count.csv")
          prob = 0.05 概率函数低于prob的任务被试为到达概率为0
    return：state 状态价值矩阵
    保存state在 "../data/state_value_dim2.csv"
    """
    if len(task_prob_float) == 0:
        task_prob_float_file = str(
            "../srcData/state_value/" + filename + "/task_prob_float.csv")
        task_prob_float = np.loadtxt(task_prob_float_file, delimiter=",")
    if len(smaller_task_count_float) == 0:
        smaller_task_count_float_file = str(
            "../srcData/state_value/" + filename + "/smaller_task_count_float.csv")
        smaller_task_count_float = np.loadtxt(smaller_task_count_float_file, delimiter=",")
    state_float = np.zeros(state_size)
    state_float[0][0] = 0
    size1 = len(task_prob_float) - 1 #cpu
    size2 = len(task_prob_float[0]) - 1 #gpu
    pbar = tqdm(total=len(task_prob_float))
    if prob == 0:
        prob = 1.0 / smaller_task_count_float[size1][size2]
    for j in range(1,state_size[1]):
        if beta_F(smaller_task_count_float, np.array([0, j]),sum) < prob:
            state_float[0][j] = j * 0.1 * ParamHolder().cpu_gpu_rate
        else:
            temp = 0
            for u_j in range(max(j-size2,0),j):
                temp += state_float[0][u_j] * beta_f_F(task_prob_float, smaller_task_count_float, np.array([0, j - u_j]), np.array([0, j]))
            state_float[0][j] = temp
    pbar.update(1)
    for i in range(1,state_size[0]):
        for j in range(0,state_size[1]):
            if beta_F(smaller_task_count_float, np.array([i, j]),sum) < prob:
                state_float[i][j] = i + j * 0.1 * ParamHolder().cpu_gpu_rate
            else:
                temp = 0
                for u_i in range(max(i-size1,0),i):
                    for u_j in range(max(j-size2,0),j+1):
                        temp += state_float[u_i][u_j] * beta_f_F(task_prob_float, smaller_task_count_float, np.array([i - u_i, j - u_j]), np.array([i, j]))
                for u_j in range(max(i-size2,0),j):
                    temp += state_float[i][u_j] * beta_f_F(task_prob_float, smaller_task_count_float, np.array([0, j - u_j]), np.array([i, j]))
                state_float[i][j] = temp
        pbar.update(1)
    pbar.close()
    np.savetxt("../srcData/state_value/" + filename + "/state_float.csv", state_float, delimiter=",")
    return state_float


def state_value_int(sum, task_prob_int = [], smaller_task_count_int = [], state_float = [],state_only_float =[], prob = 0.05, state_size = (97, 10), filename = None):
    """
    计算二维状态下的状态价值
    args： state_size = (96,96) 状态空间的大小
          task_prob_file = str("../data/task_prob.csv")
          smaller_task_count_file = str("../data/smaller_task_count.csv")
          prob = 0.05 概率函数低于prob的任务被试为到达概率为0
    return：state 状态价值矩阵
    保存state在 "../data/state_value_dim2.csv"
    """
    if len(task_prob_int) == 0:
        task_prob_file_int = str(
            "../srcData/state_value/" + filename + "/task_prob_int.csv")
        task_prob_int = np.loadtxt(task_prob_file_int, delimiter=",")
    if len(smaller_task_count_int) == 0:
        smaller_task_count_file_int = str(
            "../srcData/state_value/" + filename + "/smaller_task_count_int.csv")
        smaller_task_count_int = np.loadtxt(smaller_task_count_file_int, delimiter=",")
    if len(state_float) == 0:
        state_float_file = str(
            "../srcData/state_value/" + filename + "/state_float.csv")
        state_float = np.loadtxt(state_float_file, delimiter=",")
    if len(state_only_float) == 0:
        state_only_float_file = str(
            "../srcData/state_value/" + filename + "/state_only_float.csv")
        state_only_float = np.loadtxt(state_only_float_file, delimiter=",")
    state_int = np.zeros(state_size)
    state_int[0][0] = 0
    size1 = len(task_prob_int) - 1 #cpu
    size2 = len(task_prob_int[0]) - 1 #gpu
    pbar = tqdm(total=len(task_prob_int))
    if prob == 0:
        prob = 1.0 / smaller_task_count_int[size1][size2]
    for j in range(1,state_size[1]):
        if beta_F(smaller_task_count_int, np.array([0, j]),sum) < prob:
            state_int[0][j] = j * ParamHolder().cpu_gpu_rate
        else:
            temp = 0
            for u_j in range(max(j-size2,0),j):
                temp += state_int[0][u_j] * beta_f_F(task_prob_int, smaller_task_count_int, np.array([0, j - u_j]), np.array([0, j]))
            for u_j in range(1,10):
                temp +=(state_only_float[u_j] + state_int[0][j-1]) * beta_f_F(task_prob_int, smaller_task_count_int, np.array([0, j - u_j]), np.array([0, j]))
            state_int[0][j] = temp
    pbar.update(1)
    for i in range(1,state_size[0]):
        for j in range(0,state_size[1]):
            if beta_F(smaller_task_count_int, np.array([i, j]),sum) < prob:
                state_int[i][j] = i + j * ParamHolder().cpu_gpu_rate
            else:
                temp = 0
                for u_i in range(max(i-size1,0),i):
                    for u_j in range(max(j-size2,0),j+1):
                        temp += state_int[u_i][u_j] * beta_f_F(task_prob_int, smaller_task_count_int, np.array([i - u_i, j - u_j]), np.array([i, j]))
                    for u_j in range(1,10):
                        temp += (state_only_float[u_j] + state_int[u_i][j-1]) * beta_f_F(task_prob_int, smaller_task_count_int, np.array([i - u_i, j - u_j]), np.array([i, j]))
                for u_j in range(max(i-size2,0),j):
                    temp += state_int[i][u_j] * beta_f_F(task_prob_int, smaller_task_count_int, np.array([0, j - u_j]), np.array([i, j]))
                for u_j in range(1, 10):
                    temp += (state_only_float[u_j] + state_int[i][j-1]) * beta_f_F(task_prob_int,smaller_task_count_int,np.array([i - u_i, j - u_j]),np.array([i, j]))
                state_int[i][j] = temp
        pbar.update(1)
    pbar.close()
    np.savetxt("../srcData/state_value/" + filename + "/state_int.csv", state_int, delimiter=",")
    return state_int

def state_value_only_float(sum, task_prob_float = [], smaller_task_count_float = [], prob = 0.05, filename = None):
    if len(task_prob_float) == 0:
        task_prob_file_float = str(
            "../srcData/state_value/" + filename + "/task_prob_float.csv")
        task_prob_float = np.loadtxt(task_prob_file_float, delimiter=",")
    if len(smaller_task_count_float) == 0:
        smaller_task_count_file_float = str(
            "../srcData/state_value/" + filename + "/smaller_task_count_float.csv")
        smaller_task_count_float = np.loadtxt(smaller_task_count_file_float, delimiter=",")
    state_only_float = np.zeros(10)
    state_only_float[0] = 0
    size1 = len(task_prob_float) - 1 #cpu
    size2 = len(task_prob_float[0]) - 1 #gpu
    pbar = tqdm(total=10)
    if prob == 0:
        prob = 1.0 / smaller_task_count_float[size1][size2]
    for j in range(1,10):
        if beta_F(smaller_task_count_float, np.array([size1, j]),sum) < prob:
            state_only_float[j] = j * 0.1 * ParamHolder().cpu_gpu_rate
        else:
            temp = 0
            for u_j in range(j):
                temp += state_only_float[u_j] * beta_f_F(task_prob_float, smaller_task_count_float, np.array([size1, j - u_j]), np.array([size1, j]))
            state_only_float[j] = temp
        pbar.update(1)
        pbar.close()
    np.savetxt("../srcData/state_value/" + filename + "/state_only_float.csv", state_only_float, delimiter=",")
    return state_only_float

def generate_state(filename):
    num = cal_smaller_task_count(filename=filename)
    state_value_float(sum=int(num),state_size=(200,10),filename=filename)
    state_value_only_float(sum=int(num),filename=filename)
    state_value_int(sum=int(num),state_size=(200,9),filename=filename)

if __name__ == "__main__":
    filename = "openb_pod_list_multigpu50"
    generate_state(filename)