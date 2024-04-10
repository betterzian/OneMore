import numpy as np
from src.envSim.simParam import ParamHolder

def float_to_int(x):
    return int(x*10)

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

def test_state_value(task_prob = None,smaller_task_count = None, state_size = (97,97)):
    task_prob_file = f"../srcData/state_value/{ParamHolder().filename}/task_prob.csv"
    smaller_task_count_file = f"../srcData/state_value/{ParamHolder().filename}/smaller_task_count.csv"
    if task_prob is None:
        task_prob = np.loadtxt(task_prob_file, delimiter=",")
    if smaller_task_count is None:
        smaller_task_count = np.loadtxt(smaller_task_count_file, delimiter=",")
    prob = ParamHolder().prob * 1.0 / 100
    summary = smaller_task_count[-1][-1]
    state = np.zeros(state_size)
    state[0][0] = 0
    size = len(task_prob) - 1
    if prob == 0:
        prob = 1.0/smaller_task_count[size][size]
    for j in range(1,len(state)):
        if beta_F(smaller_task_count,np.array([0,j]),summary) < prob:
            state[0][j] = j
        else:
            temp = 0
            for u_j in range(max(j-size,0),j):
                temp += state[0][u_j] * beta_f_F(task_prob, smaller_task_count, np.array([0, j - u_j]), np.array([0, j]))
            state[0][j] = temp
    for i in range(1,len(state)):
        for j in range(0,len(state)):
            if beta_F(smaller_task_count,np.array([i,j]),summary) < prob:
                state[i][j] = i + j
            else:
                temp = 0
                for u_i in range(max(i-size,0),i):
                    for u_j in range(max(j-size,0),j+1):
                        temp += state[u_i][u_j] * beta_f_F(task_prob, smaller_task_count, np.array([i - u_i, j - u_j]), np.array([i, j]))
                for u_j in range(max(i-size,0),j):
                    temp += state[i][u_j] * beta_f_F(task_prob, smaller_task_count, np.array([0, j - u_j]), np.array([i, j]))
                state[i][j] = temp
    np.savetxt(f"../srcData/state_value/{ParamHolder().filename}/state{ParamHolder().prob}.csv", state, delimiter=',')
    return state

def cal_smaller_task_count(task_prob = None):
    if task_prob is None:
        task_prob_file = f"../srcData/state_value/{ParamHolder().filename}/task_prob.csv"
        task_prob = np.loadtxt(task_prob_file,delimiter=',')
    smaller_task_count = np.zeros(task_prob.shape)
    smaller_task_count[0][0] = task_prob[0][0]
    for i in range(1, len(smaller_task_count)):
        smaller_task_count[i][0] = smaller_task_count[i - 1][0] + task_prob[i][0]
        smaller_task_count[0][i] = smaller_task_count[0][i - 1] + task_prob[0][i]
    for i in range(1, len(smaller_task_count)):
        for j in range(1, len(smaller_task_count)):
            smaller_task_count[i][j] = smaller_task_count[i][j - 1] + smaller_task_count[i - 1][j] - smaller_task_count[i - 1][j - 1] + \
                                   task_prob[i][j]
    np.savetxt(f"../srcData/state_value/{ParamHolder().filename}/smaller_task_count.csv", smaller_task_count, delimiter=',')
    return smaller_task_count

def generate_state():
    cal_smaller_task_count()
    test_state_value()

if __name__ == "__main__":
    generate_state()