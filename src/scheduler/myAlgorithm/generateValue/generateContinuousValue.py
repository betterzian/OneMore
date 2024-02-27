import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn import datasets
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


class TrainBot():
    def __init__(self,lr = 0.01,epoch = 1000):
        self.__lr = lr
        self.__epoch = epoch
        self.__device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.__state_value = StateValue(9).to(self.__device)
        self.__optimizer = torch.optim.Adam(self.__state_value.parameters(), lr=self.__lr)
        self.__criterion = nn.MSELoss()
        self.__task_list = np.loadtxt("../data_src/offline_task/off_task_list.csv", delimiter=',', dtype=float)
        self.__task_len = len(self.__task_list)
        self.__cpu_gpu_rate = 10

    def train(self):
        for i in range(self.__epoch):
            state, state_sum = self.__generate_state()
            task_list = self.__generate_task()
            next_value = 0
            state_len = []
            next_state = []
            with torch.no_grad():
                for task in task_list:
                    temp_next = self.__get_next_state(state, task)
                    if len(temp_next) == 0:
                        next_value += state_sum
                    else:
                        next_state.extend(temp_next)
                        state_len.append(len(temp_next))
                if len(next_state) != 0:
                    next_state = np.array(next_state)
                    next_state_out = self.__state_value(torch.tensor(next_state, dtype=torch.float32).to(self.__device))
                    next_state_out = [torch.max(elem) for elem in torch.split(next_state_out, state_len)]
                    next_value += sum(next_state_out)
                next_value /= 100.0
                next_value = torch.tensor([next_value],dtype=torch.float32).to(self.__device)
            state = torch.tensor(state,dtype=torch.float32).to(self.__device)
            value = self.__state_value(state)
            loss = self.__criterion(next_value, value)
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()

    def __get_next_state(self, state, task):
        if state[0] < task[0]:
            return []
        if task[1] < 1:
            diag_matrix = np.diag(np.ones(8) * task[1])
            next_state = state[1:] - diag_matrix
            positive_cols = np.all(next_state >= 0, axis=0)
            next_state = next_state[positive_cols, :]
            next_state = abs(np.sort(-next_state, axis=1))
            next_state = np.insert(next_state, 0, np.ones(len(next_state)) * (state[0] - task[0]), axis=1)
            return next_state.reshape(-1, 9)
        cpu = state[0]
        state[0] = 100
        next_state = abs(np.sort(-state))
        for i in range(int(task[1])):
            next_state[i + 1] -= 1
            if next_state[i + 1] < 0:
                state[0] = cpu
                return []
        next_state = abs(np.sort(-next_state))
        next_state[0] = cpu - task[0]
        state[0] = cpu
        return next_state.reshape(-1, 9)

    def __generate_state(self, cpu_max=128, gpu_max=11):
        state_sum = 0
        state = np.random.randint(0, gpu_max, 9)
        state = np.round(state / 10.0, 1)
        state_sum += state.sum() * self.__cpu_gpu_rate
        state[0] = 100
        state = abs(np.sort(-state))
        state[0] = round(random.uniform(0, cpu_max), 1)
        state_sum += state[0]
        return state,state_sum

    def __generate_task(self):
        site = np.random.randint(0, self.__task_len, 100)
        return self.__task_list[site]


class MemPool():
    def __init__(self, max_len, batch_size):
        self.__mem = []
        self.__len = 0
        self.__max_len = max_len
        self.__batch_size = batch_size

    def add_mem(self, mem):
        pass

    def get_mem(self):
        pass


if __name__ == "__main__":
    train_bot = TrainBot()
    train_bot.train()
