import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random

class StateValue(nn.Module):
    def __init__(self, state_dim):
        super(StateValue, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.sp = nn.Softplus()

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        return self.sp(x)


def train():
    pass


def get_next_state(state,task):
    if state[0] < task[0]:
        return False
    if task[1] < 1:
        diag_matrix = np.diag(np.ones(8)*task[1])
        next_state = state[:,1:] - diag_matrix
        positive_cols = np.all(next_state >= 0, axis=0)
        next_state = next_state[positive_cols,:]
        next_state = np.sort(next_state, axis=1)
        next_state = np.insert(np.ones((8, 1)) * (state[0] - task[0]), 1, next_state, axis=1)
        return next_state
    cpu = state[0]
    state[0] = 100
    next_state = np.sort(state, axis=1)
    for i in range(int(task[1])):
        next_state[i+1] -= 1
        if next_state[i+1] < 0:
            return False
    next_state = np.sort(next_state, axis=1)
    next_state[0] = cpu - task[0]
    state[0] = cpu
    return next_state

class MemPool():
    def __init__(self,max_len,batch_size):
        self.__mem = []
        self.__len = 0
        self.__max_len = max_len
        self.__batch_size = batch_size
    def add_mem(self,mem):
        pass

    def get_mem(self):
        pass



if __name__ == "__main__":
    lr = 0.01
    epoch = 1000
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    state_value = StateValue(9).to(device)
    optimizer = torch.optim.Adam(state_value.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for i in range(epoch):
        state = generate_state()
        task_list = generate_task()
        next_value = 0
        state_len = []
        next_state = []
        with torch.no_grad():
            for task in task_list:
                temp_next = get_next_state(state,task)
                if not temp_next:
                    temp_next += state.sum()
                else:
                    next_state.append(temp_next)
                    state_len.append(len(temp_next))
            next_state = np.array(next_state)
            next_value = state_value(torch.from_numpy(next_state))
            next_state = [torch.max(elem) for elem in torch.split(next_state,state_len)]
            next_value += sum(next_state)
        state = torch.from_numpy(state)
        value = state_value(state)
        loss = criterion(next_value, value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

