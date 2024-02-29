import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
import math
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from src.envSim.simParam import ParamHolder

class StateValue(nn.Module):
    def __init__(self, state_dim):
        super(StateValue, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 1)

        for name, param in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.normal_(param, mean=0, std=0.01)
            elif 'bias' in name:
                torch.nn.init.normal_(param, mean=0, std=0.01)

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)
        self.sp = nn.Softplus()

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.leaky_relu(self.fc4(x))
        x = self.dropout3(x)
        x = F.leaky_relu(self.fc5(x))
        return self.sp(x)

class BufferArray:
    def __init__(self, memo_max_len,task_num=100,state_dim=9):
        self.__state = np.ones((memo_max_len, state_dim), dtype=np.float32) * -1
        self.__task_len = np.ones((memo_max_len, task_num)).astype(int) * -1
        self.__next_state = np.ones((memo_max_len, task_num, state_dim-1, state_dim), dtype=np.float32) * -1
        self.__state_value = np.zeros(memo_max_len, dtype=np.float32)
        self.__task_num = task_num
        self.__state_dim = state_dim
        self.__next_idx = 0
        self.__is_full = False
        self.__max_len = memo_max_len
        self.__now_len = self.__max_len if self.__is_full else self.__next_idx

    def add_memo(self, state=None, next_state=None, state_len=None, state_value=None, empty=False):
        if empty:
            self.__next_idx += 1
        else:
            self.__state[self.__next_idx] = state.reshape(-1,self.__state_dim)
            self.__task_len[self.__next_idx] = state_len.reshape(-1,self.__task_num)
            self.__next_state[self.__next_idx] = next_state.reshape(-1,self.__task_num, self.__state_dim-1, self.__state_dim)
            self.__state_value[self.__next_idx] = state_value
            self.__next_idx = self.__next_idx + 1
        if self.__next_idx >= self.__max_len:
            self.__is_full = True
            self.__next_idx = 0
        self.__now_len = self.__max_len if self.__is_full else self.__next_idx

    def random_sample(self, batch_size, device):
        if batch_size == -1:
            batch_size = self.__now_len
        if batch_size > self.__now_len:
            batch_size = self.__now_len
        indices = np.random.randint(self.__now_len, size=batch_size)
        state = self.__state[indices]
        state_len = self.__task_len[indices]
        next_state = self.__next_state[indices]
        state_value = self.__state_value[indices]
        task_num = state_value
        task_num = task_num.reshape(-1).astype(int)
        state_len = state_len.reshape(-1)
        state_len = state_len[state_len > 0]
        next_state = next_state.reshape(-1,self.__state_dim)
        next_state = next_state[next_state[:,0] != -1]
        state = torch.tensor(state, device=device)
        next_state = torch.tensor(next_state, device=device)
        return state,next_state,state_len,task_num,state_value


class TrainBot():
    def __init__(self,lr = 0.001,epoch = 2000):
        self.__lr = lr
        self.__epoch = epoch
        self.__device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.__state_value = StateValue(9).to(self.__device)
        self.__state_value.train()
        if os.path.exists("../srcData/offline_task/"+ParamHolder().filename+"_model/model.pth"):
            self.__state_value.load_state_dict(torch.load("../srcData/offline_task/"+ParamHolder().filename+"_model/model.pth"))
        self.__optimizer = torch.optim.Adam(self.__state_value.parameters(), lr=self.__lr)
        self.__criterion = nn.MSELoss()
        self.__task_list = np.loadtxt("../srcData/state_value/"+ParamHolder().filename+"/off_task_list.csv", delimiter=',', dtype=float)
        self.__task_len = len(self.__task_list)
        self.__cpu_gpu_rate = ParamHolder().cpu_gpu_rate
        self.__reply_buffer = BufferArray(409600)
        self.__reply_buffer_with_real_state = BufferArray(409600)
        self.__writer = SummaryWriter("../log/"+ParamHolder().filename+"_model")

    def train(self):
        pbar = tqdm(total=self.__epoch,desc="epoch")
        turn = 0
        for i in range(self.__epoch):
            loss = 0
            for count in range(1,2001):
                state, state_sum = self.__generate_state(i)
                task_list = self.__generate_task()
                state_len = np.ones(100).astype(int) * -1
                temp_len = np.ones(100).astype(int) * -1
                next_state = np.ones((100,8,9), dtype=np.float32) * -1
                state_value = 0
                for j in range(len(task_list)):
                    temp_next = self.__get_next_state(state, task_list[j])
                    state_value = state_sum
                    if temp_next is not None:
                        state_len[j] = len(temp_next)
                        next_state[j,0:state_len[j],:] = temp_next
                loc = state_len > 0
                state_len = state_len[loc]
                if len(state_len) > 0:
                    state_value = len(state_len)
                    temp_len[:len(state_len)] = state_len
                    self.__reply_buffer.add_memo(state, next_state, temp_len, state_value)
                else:
                    self.__reply_buffer_with_real_state.add_memo(state, next_state, temp_len, state_value)
                if count % 500 == 0:
                    loss += self.update(count)
                    pbar.set_description(f"Loss: {loss/(int(count/100) + 1)}")
                    self.__writer.add_scalar("avg_loss", loss/(int(count/100) + 1), turn)
                    turn += 1
            pbar.update(1)
            torch.save(self.__state_value.state_dict(), "../srcData/offline_task/"+ParamHolder().filename+"_model/model.pth")
        pbar.close()
        torch.save(self.__state_value.state_dict(), "../srcData/offline_task/"+ParamHolder().filename+"_model/model.pth")

    def update(self,count):
        epoch = 6 - int(count/500)
        avg_loss = 0
        for _ in range(epoch):
            state,next_state,state_len,task_num,_ = self.__reply_buffer.random_sample(512,self.__device)
            state_real,_,_,_,state_value_real = self.__reply_buffer_with_real_state.random_sample(512,self.__device)
            state = torch.cat((state, state_real), dim=0)

            with torch.no_grad():
                state_value = torch.tensor(state_value_real, dtype=torch.float32).to(self.__device).reshape(-1, 1)
                if len(next_state) != 0:
                    next_state_out = self.__state_value(next_state)
                    next_state_out = torch.tensor([torch.min(elem) for elem in torch.split(next_state_out, state_len.tolist())])
                    next_state_out = [torch.mean(elem) for elem in torch.split(next_state_out, task_num.tolist())]
                    next_state_out = torch.tensor(next_state_out, dtype=torch.float32).to(self.__device).reshape(-1, 1)
                    state_value = torch.cat((next_state_out, state_value), dim=0)

            value = self.__state_value(state)
            loss = self.__criterion(value, state_value)
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()
            avg_loss += abs(value.mean() - state_value.mean())
        return avg_loss/epoch

    def __get_next_state(self, state, task):
        return get_next_state(state, task)

    def __generate_state(self, i, cpu_max=129, gpu_max=10):
        gpu_max = int(gpu_max * min(1.0 , math.tanh(2 * i / self.__epoch))) + 1
        #gpu_max = gpu_max - int(gpu_max * ((self.__epoch - i) + 1) / (self.__epoch + 2))
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


def get_next_state(state, task):
    if state[0] < task[0]:
        return None
    if task[1] < 1:
        diag_matrix = np.diag(np.ones(8) * task[1])
        next_state = state[1:] - diag_matrix
        positive_cols = np.all(next_state >= 0, axis=0)
        next_state = next_state[positive_cols, :]
        next_state = abs(np.sort(-next_state, axis=1))
        next_state = np.insert(next_state, 0, np.ones(len(next_state)) * (state[0] - task[0]), axis=1)
        next_state = np.unique(next_state, axis=0)
        if len(next_state) == 0:
            return None
        return next_state.reshape(-1, 9)
    cpu = state[0]
    state[0] = 100
    next_state = abs(np.sort(-state))
    for i in range(int(task[1])):
        next_state[i + 1] -= 1
        if next_state[i + 1] < 0:
            state[0] = cpu
            return None
    next_state = abs(np.sort(-next_state))
    next_state[0] = cpu - task[0]
    state[0] = cpu
    return next_state.reshape(-1, 9)

if __name__ == "__main__":
    train_bot = TrainBot()
    train_bot.train()
