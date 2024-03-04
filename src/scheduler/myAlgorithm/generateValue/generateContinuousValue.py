import torch
import os
path = "../../../.."
path = os.path.abspath(path)
import sys
sys.path.append(path)
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
import math
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import json


class Gating(nn.Module):
    def __init__(self, input_dim, num_experts, dropout_rate=0.1):
        super(Gating, self).__init__()

        # Layers
        self.layer1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(128, 256)
        self.leaky_relu1 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer3 = nn.Linear(256, 128)
        self.leaky_relu2 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        self.layer4 = nn.Linear(128, num_experts)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.leaky_relu1(x)
        x = self.dropout2(x)
        x = self.layer3(x)
        x = self.leaky_relu2(x)
        x = self.dropout3(x)
        return torch.softmax(self.layer4(x), dim=1)

class StateValueExpert(nn.Module):
    def __init__(self, state_dim):
        super(StateValueExpert, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.dropout1 = nn.Dropout(p=0.5)
        self.sp = nn.Softplus()
        self.initialize_weights()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return self.sp(x)
    def initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.normal_(param, mean=0, std=0.01)
            elif 'bias' in name:
                torch.nn.init.normal_(param, mean=0, std=0.01)

class StateValueExpert2(nn.Module):
    def __init__(self, state_dim):
        super(StateValueExpert2, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 1)
        self.dropout1 = nn.Dropout(p=0.5)
        self.sp = nn.Softplus()
        self.initialize_weights()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return self.sp(x)

    def initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.normal_(param, mean=0, std=0.01)
            elif 'bias' in name:
                torch.nn.init.normal_(param, mean=0, std=0.01)

class MoE(nn.Module):
    def __init__(self, trained_experts):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList(trained_experts)
        num_experts = len(trained_experts)
        # Assuming all experts have the same input dimension
        input_dim = trained_experts[0].layer1.in_features
        self.gating = Gating(input_dim, num_experts)

    def forward(self, x):
        # Get the weights from the gating network
        weights = self.gating(x)

        # Calculate the expert outputs
        outputs = torch.stack([expert(x) for expert in self.experts], dim=2)

        # Adjust the weights tensor shape to match the expert outputs
        weights = weights.unsqueeze(1).expand_as(outputs)

        # Multiply the expert outputs with the weights and
        # sum along the third dimension
        return torch.sum(outputs * weights, dim=2)




class BufferArray:
    def __init__(self, memo_max_len,task_num=100,state_dim=9):
        self.__state = np.ones((memo_max_len,state_dim), dtype=np.float32) * -1
        self.__sin_task_len = np.ones((memo_max_len, task_num)).astype(int) * -1
        self.__next_state = np.ones((memo_max_len, task_num, state_dim-1, state_dim), dtype=np.float32) * -1
        self.__state_value = np.zeros(memo_max_len, dtype=np.float32)
        self.__expert_mark = np.ones((memo_max_len,task_num,state_dim-1),dtype=np.float32) * -1
        self.__task_num = task_num
        self.__state_dim = state_dim
        self.__next_idx = 0
        self.__is_full = False
        self.__max_len = memo_max_len
        self.__now_len = self.__max_len if self.__is_full else self.__next_idx

    def init_again(self):
        self.__next_idx = 0
        self.__is_full = False
        self.__now_len = self.__max_len if self.__is_full else self.__next_idx

    def add_memo(self, state=None, next_state=None, sin_task_len=None, state_value=None, expert_mark=None, empty=False):
        if empty:
            self.__next_idx += 1
        else:
            self.__state[self.__next_idx] = state.reshape(-1,self.__state_dim)
            self.__state_value[self.__next_idx] = state_value
            if state_value < 0:
                self.__sin_task_len[self.__next_idx] = sin_task_len.reshape(-1, self.__task_num)
                self.__next_state[self.__next_idx] = next_state.reshape(-1,self.__task_num, self.__state_dim-1, self.__state_dim)
                self.__expert_mark[self.__next_idx] = expert_mark.reshape(-1,self.__task_num, self.__state_dim-1)
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
        sin_task_len = self.__sin_task_len[indices]
        next_state = self.__next_state[indices]
        state_value = self.__state_value[indices]
        expert_mark = self.__expert_mark[indices]

        state_value = state_value.reshape(-1)
        sin_task_len = sin_task_len.reshape(-1)
        sin_task_len = sin_task_len[sin_task_len > 0]
        next_state = next_state.reshape(-1,self.__state_dim)
        expert_mark = expert_mark.reshape(-1)
        indices = next_state[:,0] != -1
        next_state = next_state[indices]
        expert_mark = expert_mark[indices]
        state = torch.tensor(state, device=device)
        next_state = torch.tensor(next_state, device=device)
        return state,next_state,sin_task_len,state_value,expert_mark


class TrainBot():
    def __init__(self,filename,lr = 0.0005,epoch = 2000):
        self.__lr = lr
        self.__filename = filename
        self.__epoch = epoch
        self.__device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.__now_expect = 0
        self.__trained_experts =[]
        if not os.path.exists(path + f"/srcData/offline_task/model/{self.__filename}_model"):
            os.mkdir(path + f"/srcData/offline_task/model/{self.__filename}_model")
        for i in range(12):
            self.__trained_experts.append(StateValueExpert(9).to(self.__device))
            if os.path.exists(path + f"/srcData/offline_task/model/{self.__filename}_model/model{i}.pth"):
                self.__trained_experts[i].load_state_dict(torch.load(path + f"/srcData/offline_task/model/{self.__filename}_model/model{i}.pth"))
        # for i in range(10):
        #     self.__trained_experts.append(StateValueExpert2(9).to(self.__device))
        #     if os.path.exists(path + f"/srcData/offline_task/model/{self.__filename}_model/model{i}.pth"):
        #         self.__trained_experts[i].load_state_dict(
        #             torch.load(path + f"/srcData/offline_task/model/{self.__filename}_model/model{i}.pth"))
        # self.__state_value = MoE(self.__trained_experts).to(self.__device)
        # self.__state_value.train()
        self.__target_net = StateValueExpert(9).to(self.__device)
        self.__optimizer = torch.optim.Adam(self.__target_net.parameters(),lr=self.__lr)
        # if os.path.exists(path+f"/srcData/offline_task/model/{self.__filename}_model/model.pth"):
        #     self.__state_value.load_state_dict(torch.load(path+f"/srcData/offline_task/model/{self.__filename}_model/model.pth"))
        # self.__moe_optimizers = torch.optim.Adam(self.__state_value.parameters(), lr=self.__lr)
        self.__criterion = nn.HuberLoss()
        self.__task_list = np.loadtxt(path+f"/srcData/state_value/{self.__filename}/off_task_list.csv", delimiter=',', dtype=float)
        self.__task_len = len(self.__task_list)
        self.__cpu_gpu_rate = self.__get_cgr(self.__filename)
        self.__tl_one_time = 100
        self.__reply_buffer = BufferArray(409600,task_num=self.__tl_one_time)
        self.__writer = SummaryWriter(path+f"/log/{self.__filename}_model")
        self.__loss = 100
        self.__count = 1

    def train_expert(self):
        pbar = tqdm(total=self.__epoch,desc=f"{self.__filename} epoch")
        turn = 0
        while self.__now_expect < 12:
            loss = 0
            for count in range(1,2001):
                state, state_sum = self.__generate_state()
                task_list = self.__generate_task()
                state_len = np.ones(self.__tl_one_time).astype(int) * -1
                temp_len = np.ones(self.__tl_one_time).astype(int) * -1
                next_state = np.ones((self.__tl_one_time,8,9), dtype=np.float32) * -1
                expert_mark = np.zeros((self.__tl_one_time,8)).astype(int)
                state_value = 0
                for j in range(len(task_list)):
                    temp_next = self.__get_next_state(state, task_list[j])
                    state_value = state_sum
                    if temp_next is not None:
                        state_len[j] = len(temp_next)
                        expert_mark[j, 0:state_len[j]] = self.__get_expert_num(temp_next)
                        next_state[j,0:state_len[j],:] = deal_state(temp_next,self.__cpu_gpu_rate)
                loc = state_len > 0
                state_len = state_len[loc]
                if len(state_len) > 4:
                    state_value = -len(state_len)
                    temp_len[:len(state_len)] = state_len
                self.__reply_buffer.add_memo(deal_state(np.expand_dims(state,axis=0),self.__cpu_gpu_rate), next_state, temp_len, state_value, expert_mark)
                if count % 500 == 0:
                    loss += self.update_expert(count)
                    pbar.set_description(f"gpu_max: {self.__now_expect} and Loss: {loss/(int(count/500) + 1)}")
                if count % 2000 == 0:
                    self.__loss = loss / (int(count / 500) + 1)
                    self.__writer.add_scalar("avg_loss", loss / (int(count / 500) + 1), turn)
                    turn += 1
                    if turn % 5 == 0:
                        self.__trained_experts[self.__now_expect].load_state_dict(self.__target_net.state_dict())
            if self.__loss < 0.1:
                self.__count += 1
            if self.__count % 21 == 0:
                self.__count = 1
                self.__loss = 100
                self.__trained_experts[self.__now_expect].load_state_dict(self.__target_net.state_dict())
                self.__target_net.initialize_weights()
                self.__reply_buffer.init_again()
                torch.save(self.__trained_experts[self.__now_expect].state_dict(), path+f"/srcData/offline_task/model/{self.__filename}_model/model{self.__now_expect}.pth")
                self.__now_expect += 1
            pbar.update(1)
        pbar.close()

    def update_expert(self,count):
        epoch = 5 + 2 * int(count/500)
        avg_loss = 0
        for _ in range(epoch):
            state,next_state,sin_task_len,state_value,expert_mark = self.__reply_buffer.random_sample(512,self.__device)
            with torch.no_grad():
                task_num_indices = state_value < 0
                task_num = - state_value[task_num_indices].astype(int)
                state_value = torch.tensor(state_value, dtype=torch.float32).to(self.__device)
                if len(next_state) != 0:
                    next_state_out = torch.zeros(len(next_state),dtype=torch.float32).to(self.__device)
                    for i in range(len(self.__trained_experts)):
                        indices = expert_mark == i
                        temp_state = next_state[indices]
                        if len(temp_state) > 0:
                            next_state_out[indices] = self.__trained_experts[i](temp_state).reshape(-1)
                    next_state_out = torch.tensor([torch.min(elem) for elem in torch.split(next_state_out, sin_task_len.tolist())])
                    next_state_out = [torch.mean(elem) for elem in torch.split(next_state_out, task_num.tolist())]
                    next_state_out = torch.tensor(next_state_out, dtype=torch.float32).to(self.__device).reshape(-1)
                    state_value[task_num_indices] = next_state_out
            value = self.__target_net(state)
            state_value = state_value.reshape(-1, 1)
            loss = self.__criterion(value,state_value)
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()
            avg_loss += abs(value - state_value).mean()
        return avg_loss/epoch

    def __get_next_state(self, state, task):
        return get_next_state(state, task)

    def __get_cgr(self,filename):
        with open(path+"/srcData/state_value/" + filename + "/data.json", "r") as json_file:
            data = json.load(json_file)
            return data["cgr"]

    def __generate_state(self, i=0, cpu_max=129, gpu_max=11):
        gpu_max = min(self.__now_expect,gpu_max)
        # gpu_max = int(gpu_max * min(1.0, math.tanh(2 * i / self.__epoch))) + 1
        if gpu_max == 0:
            gpu_max = 11
            cpu_max = 0
        state = np.random.randint(0, gpu_max, 9)
        if cpu_max != 0:
            state[1] = gpu_max - 1
        state = np.round(state / 10.0, 1)
        state[0] = 100
        state = abs(np.sort(-state))
        state[0] = random.randint(0, cpu_max)
        state_sum = state[0] + state[1:].sum()*self.__cpu_gpu_rate
        return state,state_sum+8

    def __generate_task(self):
        site = np.random.randint(0, self.__task_len, self.__tl_one_time)
        return self.__task_list[site]

    def __get_expert_num(self,temp_next):
        expert = temp_next[:,1] + 1
        expert[temp_next[:,0] == 0] = 0
        expert = expert.astype(int)
        return expert

def get_next_state(state, task):
    if state[0] < task[0]:
        return None
    if task[1] == 0:
        next_state = state - 0
        next_state[0] -= task[0]
        return next_state.reshape(-1, 9)
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

def deal_state(state,rate):
    cpu = state[:,0] - 0
    state *= rate
    state[:,0] = cpu
    state += 1
    return state

def run(filename):
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_bot = TrainBot(filename)
    train_bot.train_expert()

if __name__ == "__main__":
    from torch import multiprocessing as mp
    filename_list = ["node","openb_pod_list_gpushare100","openb_pod_list_multigpu50"]
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(len(filename_list))
    for filename in filename_list:
        pool.apply_async(run, args=(filename,))
    pool.close()
    pool.join()
    #run("openb_pod_list_multigpu50")
