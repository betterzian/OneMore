from src.scheduler.schedulerClass import Scheduler
import numpy as np
from src.scheduler.myAlgorithm.generateValue.generateContinuousValue import StateValueExpert
from src.envSim.simParam import ParamHolder
import torch
import os


class DGGSVbyNet(Scheduler):
    def __init__(self, cluster, can_predict=True, task_mem={}, node_mem={}):
        super().__init__(cluster, can_predict, task_mem, node_mem)
        if ParamHolder().device == "cpu":
            self.__device = torch.device("cpu")
        else:
            self.__device = None
        self.agent = None
        self.__task_prob_int = None
        self.__task_prob_float = None
        self.__task_max_num = None
        self.__cpu_max_num = None
        self.__gpu_max_num = None
        self.__old_state = None
        self.__new_state = None
        self.__diag = None
        self.__indices = None
        self.__prob = None
        self.__smaller = None

    def init_again(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(ParamHolder().cuda)
        self.__device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
        self.agent = StateValueExpert(9).to(self.__device)
        self.agent.load_state_dict(
            torch.load(f"../srcData/offline_task/model/{ParamHolder().filename}_model/model.pth"))
        self.agent.eval()
        self.__smaller = np.loadtxt(f"../srcData/state_value/{ParamHolder().filename}/smaller_task_count.csv",delimiter=",")
        self.__maxcpu_array = self.__smaller.max(axis=0)
        self.__old_state = np.zeros(9)
        self.__new_state = np.zeros((8, 9))
        self.__diag = np.diag(np.ones(9))
        self.__diag = self.__diag[1:, :]
        self.__indices = np.zeros(8, dtype=int)
        self.__weight = ParamHolder().weight
        self.__prob = ParamHolder().prob * 1.0 / 100

    def release(self):
        self.agent.cpu()
        self.agent = None
        return

    def run(self, task):
        task_cpu, task_gpu = self.get_task_info(task)
        if len(task_gpu) == 0:
            task_gpu = np.array(0)
        else:
            task_gpu = task_gpu[:, 0].sum()
        now_priority = -1e8
        now_select = -1
        gpu_select = {}
        for node in self.cluster:
            temp_node_cpu, temp_node_gpu = self.get_node_info(node)
            if np.any(task_cpu > temp_node_cpu):
                continue
            else:
                self.__old_state[:] = 0
                old_state = self.__old_state
                new_state = self.__new_state
                indices = self.__indices
                temp_node_gpu = temp_node_gpu[:, 0]
                old_state[0] = temp_node_cpu.min()
                temp_new_state_0 = np.min(temp_node_cpu - task_cpu)
                old_state[1:1 + len(temp_node_gpu)] = temp_node_gpu
                if task_gpu == 0:
                    old_state = -old_state
                    old_state[1:].sort()
                    old_state = -old_state
                    new_state[0,:] = old_state.copy()
                    new_state = new_state[0]
                    new_state[0] = temp_new_state_0
                    priority, _ = self.__get_priority(old_state, new_state, 0)
                    if now_priority < priority:
                        now_priority = priority
                        gpu_select = {}
                        now_select = node
                elif task_gpu < 1:
                    new_state[:, :] = old_state - self.__diag * task_gpu
                    old_state = -old_state
                    indices[:] = np.argsort(old_state[1:])
                    old_state[1:] = old_state[1:][indices]
                    old_state = -old_state
                    new_state[:, 0] = temp_new_state_0
                    positive_cols = np.all(new_state[:, 1:] >= 0, axis=0)
                    gpu_num = positive_cols.nonzero()[0].reshape(-1)
                    new_state = new_state[positive_cols]
                    new_state = -new_state
                    new_state[:,1:].sort(axis=1)
                    new_state = -new_state
                    if new_state.size != 0:
                        priority, site = self.__get_priority(old_state, new_state, task_gpu)
                        if now_priority < priority:
                            now_priority = priority
                            gpu_select = {gpu_num[site]: 0}
                            now_select = node
                else:
                    task_gpu = int(task_gpu)
                    new_state[0,:] = old_state.copy()
                    new_state = new_state[0]
                    old_state = -old_state
                    old_state[1:].sort()
                    old_state = -old_state
                    new_state[0] = temp_new_state_0
                    if np.sum(new_state[1:] == 1) < task_gpu:
                        continue
                    indices = np.nonzero(new_state[1:] == 1)[0][:task_gpu]
                    new_state[1:][indices] = 0
                    new_state = -new_state
                    new_state[1:].sort()
                    new_state = -new_state
                    priority, _ = self.__get_priority(old_state, new_state, task_gpu)
                    if now_priority < priority:
                        now_priority = priority
                        gpu_select = dict(zip(indices, range(task_gpu)))
                        now_select = node
        if now_select != -1:
            self.set_task(now_select, task, gpu_select)
            return True
        else:
            return False

    def __get_prob(self,old_state,old_state_int,new_state,new_state_int):
        new_prob = self.__smaller[new_state[:],new_state_int[:]]
        prob = self.__smaller[old_state,old_state_int]-new_prob
        prob[self.__maxcpu_array[new_state_int[:]] < 0.01] = 0
        return prob

    def __get_bias(self,prob,old_state,new_state_int,old_state_int,int_num):
        if old_state_int == 1:
            return prob
        indices = new_state_int == old_state_int
        prob[~indices] += self.__smaller[old_state][10] - self.__smaller[old_state][int(10-10*int_num)]
        return prob

    def __get_priority(self, old_state, new_state, int_num):
        with torch.no_grad():
            new_state = new_state.reshape(-1, 9)
            new_state[:,1:] *= self._rate
            old_state[1:] *= self._rate
            if int_num == 0:
                prob = np.array(0)
            elif int_num >= 1 and np.count_nonzero(new_state[0][1:] == 10) == 0:
                prob = np.array(0)
            else:
                old_state_int = old_state[1:][(old_state[1:] == 10)]
                if old_state_int.size == 0:
                    old_state_int = np.zeros(1)
                else:
                    old_state_int = old_state_int.sum()
                if old_state_int == 0:
                    old_state_int = old_state[1:].max()
                new_state_int = (new_state[:,1:] == 10).sum(axis=1) * 10
                indices = new_state_int == 0
                new_state_int[indices] = new_state[indices, 1:].max(axis=1)
                old_state_0 = int(old_state[0])
                prob = self.__get_prob(old_state_0,old_state_int.astype(int),new_state[:,0].astype(int),new_state_int.astype(int))
                prob[prob < self.__prob] = 0
            if 0 < int_num < 1:
                prob = self.__get_bias(prob,old_state_0,new_state_int.astype(int),old_state_int.astype(int),int_num)
            old_state = torch.tensor(old_state+1,dtype=torch.float32,device=self.__device)
            new_state = torch.tensor(new_state+1,dtype=torch.float32, device=self.__device)
            old_value = self.__get_value(old_state)
            new_value = self.__get_value(new_state)
            value = old_value - new_value
            prob = torch.tensor(prob,device=self.__device)
            value -= self.__weight * prob
            max_value, max_index = torch.max(value, dim=0)
        return max_value.item(), max_index.item()

    def __get_value(self, state):
        state = state.view(-1, 9)
        return self.agent(state).view(-1)

