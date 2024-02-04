import numpy as np
def float_to_int(x):
    return int(x*10)

if __name__ == "__main__":
    src = np.loadtxt("../data_src/offline_task/node.csv", delimiter=",")
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
    np.savetxt("../data_src/offline_task/off_task_list.csv", off_task_list, delimiter=',')
    np.savetxt("../data_src/state_value/task_prob_int.csv", task_prob_int, delimiter=',')
    np.savetxt("../data_src/state_value/task_prob_float.csv", task_prob_float, delimiter=',')
