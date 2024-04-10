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
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import json
path = "../"

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

class BufferArray:
    def __init__(self, memo_max_len,task_num=100,state_dim=9):
        self.__state = np.ones((memo_max_len,state_dim), dtype=np.float32) * -1
        self.__sin_task_len = np.ones((memo_max_len, task_num)).astype(int) * -1
        self.__next_state = np.ones((memo_max_len, task_num, state_dim-1, state_dim), dtype=np.float32) * -1
        self.__state_value = np.zeros(memo_max_len, dtype=np.float32)
        self.__task_num_list = np.zeros(memo_max_len).astype(int)
        self.__expert_mark = np.ones((memo_max_len,task_num,state_dim-1),dtype=np.float32) * -1
        self.__task_num = task_num
        self.__state_dim = state_dim
        self.__next_idx = 0
        self.__is_full = False
        self.__max_len = memo_max_len
        self.__now_len = self.__max_len if self.__is_full else self.__next_idx


    def get_len(self):
        return self.__now_len

    def init_again(self):
        self.__next_idx = 0
        self.__is_full = False
        self.__now_len = self.__max_len if self.__is_full else self.__next_idx

    def add_memo(self, state=None, next_state=None, sin_task_len=None, state_value=None,task_num=None, expert_mark=None, empty=False):
        if empty:
            self.__next_idx += 1
        else:
            self.__state[self.__next_idx] = state.reshape(-1,self.__state_dim)
            self.__state_value[self.__next_idx] = state_value
            self.__task_num_list[self.__next_idx] = task_num
            if task_num > 0:
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
        state_value = self.__state_value[indices].reshape(-1)
        task_num = self.__task_num_list[indices].reshape(-1)
        expert_mark = self.__expert_mark[indices]
        indices = task_num > 0
        sin_task_len = sin_task_len[indices].reshape(-1)
        next_state = next_state[indices].reshape(-1,self.__state_dim)
        expert_mark = expert_mark[indices].reshape(-1)
        sin_task_len = sin_task_len[sin_task_len > 0]
        indices = next_state[:,0] != -1
        next_state = next_state[indices]
        expert_mark = expert_mark[indices]
        state = torch.tensor(state, device=device)
        next_state = torch.tensor(next_state, device=device)
        return state,next_state,sin_task_len,state_value,task_num,expert_mark


class TrainBot():
    def __init__(self,filename,lr = 0.0005,epoch = 2000):
        self.__lr = lr
        self.__filename = filename
        self.__epoch = epoch
        self.__device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.__now_expect = 0
        self.__task_prob_int = np.loadtxt(path + f"/srcData/state_value/{self.__filename}/smaller_task_count_int.csv", delimiter=",")
        self.__task_prob_float = np.loadtxt(path + f"/srcData/state_value/{self.__filename}/smaller_task_count_float.csv", delimiter=",")
        self.__task_max_num = self.__task_prob_int.max()
        self.__cpu_max_num = len(self.__task_prob_int) - 1
        self.__gpu_max_num = len(self.__task_prob_int[0]) - 1
        self.__trained_experts =[]
        if not os.path.exists(path + f"/srcData/offline_task/model/{self.__filename}_model"):
            os.mkdir(path + f"/srcData/offline_task/model/{self.__filename}_model")
        for i in range(10):
            self.__trained_experts.append(StateValueExpert(9).to(self.__device))
            if os.path.exists(path + f"/srcData/offline_task/model/{self.__filename}_model/model{i}.pth"):
                self.__trained_experts[i].load_state_dict(torch.load(path + f"/srcData/offline_task/model/{self.__filename}_model/model{i}.pth"))
        self.__target_net = StateValueExpert(9).to(self.__device)
        self.__optimizer = torch.optim.Adam(self.__target_net.parameters(),lr=self.__lr)
        self.__criterion = nn.HuberLoss()
        self.__task_list = np.loadtxt(path+f"/srcData/state_value/{self.__filename}/off_task_list.csv", delimiter=',', dtype=float)
        self.__task_len = len(self.__task_list)
        self.__cpu_gpu_rate = self.__get_cgr(self.__filename)
        self.__tl_one_time = 100
        self.__reply_buffer = BufferArray(40960,task_num=self.__tl_one_time)
        self.__writer = SummaryWriter(path+f"/log/{self.__filename}_model")
        self.__loss = 100
        self.__count = 1

    def train_expert(self):
        pbar = tqdm(total=self.__epoch,desc=f"{self.__filename} epoch")
        turn = 0
        while self.__now_expect < 10:
            loss = 0
            for count in range(1,2049):
                state, state_sum = self.__generate_state()
                sin_task_len = np.zeros(self.__tl_one_time).astype(int)
                temp_len = np.zeros(self.__tl_one_time).astype(int)
                next_state = np.ones((self.__tl_one_time,8,9), dtype=np.float32) * -1
                expert_mark = np.zeros((self.__tl_one_time,8)).astype(int)
                state_value = 0
                task_num = 0
                prob = self.__get_prob(state)
                if prob < 0.05:
                    state_value = state_sum
                    self.__reply_buffer.add_memo(deal_state(np.expand_dims(state,axis=0),self.__cpu_gpu_rate), next_state, temp_len, state_value,task_num, expert_mark)
                else:
                    j = 0
                    while j < 50:
                        task_list = self.__generate_task()
                        for task in task_list:
                            temp_next = self.__get_next_state(state, task)
                            if temp_next is not None:
                                sin_task_len[j] = len(temp_next)
                                expert_mark[j, 0:sin_task_len[j]] = get_expert_num(temp_next)
                                next_state[j,0:sin_task_len[j],:] = deal_state(temp_next,self.__cpu_gpu_rate)
                                j += 1
                                if j == 99:
                                    break
                    sin_task_len = sin_task_len[sin_task_len>0]
                    temp_len[:len(sin_task_len)] = sin_task_len
                    task_num = len(sin_task_len)
                    self.__reply_buffer.add_memo(deal_state(np.expand_dims(state,axis=0),self.__cpu_gpu_rate), next_state, temp_len, state_value,task_num, expert_mark)
                if count % 512 == 0:
                    loss += self.__update_expert()
                    pbar.set_description(f"gpu_max: {self.__now_expect} and Loss: {loss/(int(count/500) + 1)}")
                if count % 2048 == 0:
                    self.__loss = loss / (int(count / 500) + 1)
                    self.__writer.add_scalar("avg_loss", loss / (int(count / 500) + 1), turn)
                    turn += 1
                    if turn % 5 == 0:
                        self.__trained_experts[self.__now_expect].load_state_dict(self.__target_net.state_dict())
            if self.__loss < 1:
                self.__count += 1
            if self.__count % 200 == 0:
                self.__count = 1
                self.__loss = 100
                self.__trained_experts[self.__now_expect].load_state_dict(self.__target_net.state_dict())
                self.__target_net.initialize_weights()
                self.__reply_buffer.init_again()
                torch.save(self.__trained_experts[self.__now_expect].state_dict(), path+f"/srcData/offline_task/model/{self.__filename}_model/model{self.__now_expect}.pth")
                self.__now_expect += 1
            pbar.update(1)
        pbar.close()


    def train(self):
        self.__target_net.initialize_weights()
        self.__criterion = nn.MSELoss()
        epoch = 200000
        pbar = tqdm(total=epoch, desc=f"{self.__filename} epoch")
        avg_loss = 0
        for turn in range(epoch):
            state = self.__generate_state(batch_size=512)
            expert_mark = get_expert_num(state)
            state = deal_state(state,self.__cpu_gpu_rate)
            state = torch.tensor(state,dtype=torch.float32).to(self.__device)
            with torch.no_grad():
                real_value = torch.zeros(len(state),dtype=torch.float32).to(self.__device)
                for i in range(len(self.__trained_experts)):
                    indices = expert_mark == i
                    temp_state = state[indices]
                    if len(temp_state) > 0:
                        real_value[indices] = self.__trained_experts[i](temp_state).reshape(-1)
            value = self.__target_net(state)
            loss = self.__criterion(value,real_value.reshape(-1,1))
            self.__writer.add_scalar("loss", loss, turn)
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()
            avg_loss = abs(value - real_value.reshape(-1,1)).mean()
            pbar.set_description(f"Loss: {avg_loss}")
            pbar.update(1)
            if turn % 1000 == 0:
                torch.save(self.__target_net.state_dict(),
                           path + f"/srcData/offline_task/model/{self.__filename}_model/model.pth")
        pbar.close()

    def __update_expert(self):
        epoch = 2 + int(self.__reply_buffer.get_len() / 4096)
        avg_loss = 0
        for _ in range(epoch):
            state,next_state,sin_task_len,state_value,task_num_list,expert_mark = self.__reply_buffer.random_sample(512,self.__device)
            with torch.no_grad():
                task_num = task_num_list[task_num_list > 0]
                state_value = torch.tensor(state_value, dtype=torch.float32).to(self.__device)
                if len(next_state) != 0:
                    next_state_out = torch.zeros(len(next_state),dtype=torch.float32).to(self.__device)
                    for i in range(len(self.__trained_experts)):
                        indices = expert_mark == i
                        temp_state = next_state[indices]
                        if len(temp_state) > 0:
                            next_state_out[indices] = self.__trained_experts[i](temp_state).reshape(-1)
                    next_state_out = torch.tensor([torch.min(elem) for elem in torch.split(next_state_out, sin_task_len.tolist())])
                    next_state_out = [torch.sum(elem) for elem in torch.split(next_state_out, task_num.tolist())]
                    next_state_out = torch.tensor(next_state_out, dtype=torch.float32).to(self.__device).reshape(-1)
                    state_value[task_num_list > 0] += next_state_out
                    state_value[task_num_list > 0] /= torch.tensor(task_num_list[task_num_list > 0]).to(self.__device)
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

    def __generate_state(self, cpu_max=129, gpu_max=11, batch_size = 1):
        if batch_size == 1:
            if self.__now_expect == 0:
                cpu_max = 0
                gpu_min = 0
                num = 8
            else:
                gpu_min = 1
                num = min(self.__now_expect-1,8)
            state = np.zeros(9)
            state[1:1+num] = np.random.randint(gpu_min, gpu_max, num)
            state = np.round(state / 10.0, 1)
            state[0] = 100
            state = abs(np.sort(-state))
            state[0] = random.randint(0, cpu_max)
            state_sum = state[0] + state[1:].sum()*self.__cpu_gpu_rate
            return state,state_sum+9
        else:
            state = np.zeros((batch_size,9))
            state[:,0] = np.random.randint(0, cpu_max,size=(batch_size,))
            state[:,1:] = np.round(np.random.randint(0, gpu_max, size=(batch_size,8))/10.0,1)
            state[:,1:] = abs(np.sort(-state[:,1:],axis=1))
            return state


    def __generate_task(self):
        temp_list = np.random.choice(range(len(self.__task_list)), size=self.__tl_one_time, replace=True)
        return self.__task_list[temp_list]

    def __get_prob(self,state):
        int_num = np.count_nonzero(state == 1)
        if int_num > 0:
             return self.__task_prob_int[min(int(state[0]),self.__cpu_max_num)][min(int_num,self.__gpu_max_num)] * 1.0 / self.__task_max_num
        else:
            return self.__task_prob_float[min(int(state[0]),self.__cpu_max_num)][int(state[1] * 10) % 10] * 1.0 / self.__task_max_num


def get_expert_num(temp_next):
    expert = 9 - np.sum(temp_next[:, 1:] == 0, axis=1).reshape(-1)
    expert[temp_next[:,0] == 0] = 0
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
    state[:,1:] *= rate
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
    filename_list = ["openb_pod_list_gpushare100","openb_pod_list_multigpu50"]
    run("openb_pod_list_gpushare100")
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(len(filename_list))
    for filename in filename_list:
        pool.apply_async(run, args=(filename,))
    pool.close()
    pool.join()

