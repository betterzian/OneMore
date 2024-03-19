from src.scheduler.schedulerClass import Scheduler
import numpy as np
from src.scheduler.myAlgorithm.generateValue.generateContinuousValue import StateValueExpert, deal_state, get_expert_num
from src.envSim.simParam import ParamHolder
import torch


class ModelScheduler(Scheduler):
    def __init__(self, cluster, can_predict=True, task_mem={}, node_mem={}):
        super().__init__(cluster, can_predict, task_mem, node_mem)
        if ParamHolder().device == "cpu":
            self.__device = torch.device("cpu")
        else:
            self.__device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.__trained_experts = []
        self.__task_prob_int = torch.tensor(
            np.loadtxt(f"../srcData/state_value/{ParamHolder().filename}/smaller_task_count_int.csv",
                       delimiter=","), device=self.__device)
        self.__task_prob_float = torch.tensor(np.loadtxt(
            f"../srcData/state_value/{ParamHolder().filename}/smaller_task_count_float.csv", delimiter=","),
            device=self.__device)
        self.__task_max_num = self.__task_prob_int.max()
        self.__cpu_max_num = torch.tensor(len(self.__task_prob_int) - 1, device=self.__device)
        self.__gpu_max_num = torch.tensor(len(self.__task_prob_int[0]) - 1, device=self.__device)

        for i in range(10):
            self.__trained_experts.append(StateValueExpert(9).to(self.__device))
            self.__trained_experts[i].load_state_dict(
                torch.load(f"../srcData/offline_task/model/{ParamHolder().filename}_model/model{i}.pth"))
            self.__trained_experts[i].eval()
        # self.__buffer = BufferArray(len(cluster))

    def run(self, task):
        task_cpu, task_gpu = self.get_task_info(task)
        if len(task_gpu) == 0:
            task_gpu = torch.tensor(0, device=self.__device)
        else:
            task_gpu = torch.tensor(np.sum(task_gpu[:, 0]), device=self.__device)
        now_priority = -1e8
        now_select = -1
        gpu_select = {}
        for node in self.cluster:
            temp_node_cpu, temp_node_gpu = self.get_node_info(node)
            if np.any(task_cpu > temp_node_cpu):
                continue
            else:
                old_state = torch.zeros(9, device=self.__device)
                temp_node_gpu = torch.tensor(temp_node_gpu[:, 0], device=self.__device)
                old_state[0] = temp_node_cpu.min()
                temp_new_state_0 = np.min(temp_node_cpu - task_cpu)
                old_state[1:1 + len(temp_node_gpu)] = temp_node_gpu
                sorted_values, sorted_indices = torch.sort(temp_node_gpu, descending=True)
                if task_gpu == 0:
                    old_state[1:1 + len(sorted_values)] = sorted_values
                    new_state = torch.clone(old_state)
                    new_state[0] = temp_new_state_0
                    priority, _ = self.__get_priority(old_state, new_state, 0)
                    if now_priority < priority:
                        now_priority = priority
                        gpu_select = {}
                        now_select = node
                elif task_gpu < 1:
                    new_state = old_state - torch.diag(torch.ones(9, device=self.__device) * task_gpu)
                    old_state[1:1 + len(sorted_values)] = sorted_values
                    new_state = new_state[1:, :]
                    new_state[:, 0] = temp_new_state_0
                    positive_cols = torch.all(new_state[:, 1:] >= 0, dim=0)
                    gpu_num = positive_cols.nonzero().view(-1).cpu().numpy()
                    # gpu_num = positive_cols.nonzero().reshape(-1).numpy()
                    new_state = new_state[positive_cols, :]
                    if new_state.numel() != 0:
                        priority, site = self.__get_priority(old_state, new_state, 0.5)
                        if now_priority < priority:
                            now_priority = priority
                            gpu_select = {gpu_num[site]: 0}
                            now_select = node
                else:
                    new_state = old_state.clone()
                    old_state[1:1 + len(sorted_values)] = sorted_values
                    new_state[0] = temp_new_state_0
                    if torch.sum(new_state[1:] == 1) < task_gpu:
                        continue
                    indices = torch.nonzero(new_state[1:] == 1)[:int(task_gpu)]
                    new_state[1:][indices] = 0
                    sorted_values, sorted_indices = torch.sort(new_state[1:], descending=True)
                    new_state[1:1 + len(sorted_values)] = sorted_values
                    priority, _ = self.__get_priority(old_state, new_state, int(task_gpu))
                    if now_priority < priority:
                        now_priority = priority
                        indices = indices.view(-1).cpu().numpy()
                        gpu_select = {index: value for value, index in enumerate(indices)}
                        now_select = node

                # old_state = np.zeros(9)
                # old_state[0] = temp_node_cpu.min()
                # temp_new_state_0 = temp_node_cpu - task_cpu
                # temp_new_state_0 = temp_new_state_0.min()
                # old_state[1:1 + len(temp_node_gpu)] = abs(np.sort(-temp_node_gpu))
                # task_gpu = task_gpu.sum()
                # if task_gpu == 0:
                #     new_state = old_state - 0
                #     new_state[0] = temp_new_state_0
                #     priority,_ = self.__get_priority(old_state, new_state,0)
                #     if now_priority < priority:
                #         now_priority = priority
                #         gpu_select = {}
                #         now_select = node
                # elif task_gpu < 1:
                #     gpu_list = []
                #     gpu_num = []
                #     for i in range(len(temp_node_gpu)):
                #         temp_new_node_gpu = temp_node_gpu.copy()
                #         new_state = np.zeros(9)
                #         if temp_node_gpu[i] >= task_gpu:
                #             temp_new_node_gpu[i] -= task_gpu
                #             new_state[0] = temp_new_state_0
                #             new_state[1:1 + len(temp_new_node_gpu)] = abs(np.sort(-temp_new_node_gpu))
                #             gpu_list.append(new_state)
                #             gpu_num.append(i)
                #     if len(gpu_list) > 0:
                #         new_state = np.array(gpu_list)
                #         priority,site = self.__get_priority(old_state,new_state,0.5)
                #         if now_priority  < priority:
                #             now_priority = priority
                #             gpu_select ={gpu_num[site]: 0}
                #             now_select = node
                # else:
                #     j = 0
                #     new_state = np.zeros(9)
                #     new_state[0] = temp_new_state_0
                #     temp_task_gpu = task_gpu
                #     for i in range(len(temp_node_gpu)):
                #         if temp_task_gpu > 0 and temp_node_gpu[i] == 1:
                #             temp_task_gpu -= 1
                #             temp_node_gpu[i] -= 1
                #             temp_select[i] = j
                #             j += 1
                #     if len(temp_select) == int(task_gpu):
                #         new_state[1:1 + len(temp_node_gpu)] = abs(np.sort(-temp_node_gpu))
                #         priority,_ = self.__get_priority(old_state,new_state,int(task_gpu))
                #         if now_priority < priority:
                #             now_priority = priority
                #             gpu_select = temp_select
                #             now_select = node

        if now_select != -1:
            self.set_task(now_select, task, gpu_select)
            return True
        else:
            return False

    def __get_priority(self, old_state, new_state, int_num):
        with (((torch.no_grad()))):
            if int_num == 0:
                prob = 0
            elif int_num >= 1:
                old_int_num = torch.count_nonzero(old_state[1:] == 1)
                new_int_num = old_int_num - int_num
                if new_int_num == 0:
                    prob = 0
                else:
                    old_prob = self.__task_prob_int[torch.min(torch.floor(old_state[0]), self.__cpu_max_num).to(torch.int)][
                                   torch.min(old_int_num, self.__gpu_max_num).to(torch.int)] * 1.0 / self.__task_max_num
                    new_prob = self.__task_prob_int[torch.min(torch.floor(new_state[0]), self.__cpu_max_num).to(torch.int)][
                                   torch.min(new_int_num, self.__gpu_max_num).to(torch.int)] * 1.0 / self.__task_max_num
                    prob = old_prob - new_prob
            else:
                prob = torch.zeros(len(new_state), device=self.__device)
                old_int_num = torch.count_nonzero(old_state[1:] == 1)
                new_int_num = torch.count_nonzero(new_state[:, 1:] == 1, dim=1)
                indices = new_int_num == old_int_num
                temp_new_state = new_state[~indices]
                new_int_num = new_int_num[~indices]
                temp_list = torch.zeros(len(temp_new_state), device=self.__device)
                for i in range(len(temp_new_state)):
                    if new_int_num[i] == 0:
                        old_prob = self.__task_prob_int[torch.min(torch.floor(old_state[0]), self.__cpu_max_num).to(torch.int)][
                                       torch.min(old_int_num, self.__gpu_max_num).to(torch.int)] * 1.0 / self.__task_max_num
                        new_prob = \
                        self.__task_prob_float[torch.min(torch.floor(temp_new_state[i, 0]), self.__cpu_max_num).to(torch.int)][
                            9] * 1.0 / self.__task_max_num
                        temp_list[i] = (old_prob - new_prob)
                    else:
                        old_prob = self.__task_prob_int[torch.min(torch.floor(old_state[0]), self.__cpu_max_num).to(torch.int)][
                                       torch.min(old_int_num, self.__gpu_max_num).to(torch.int)] * 1.0 / self.__task_max_num
                        new_prob = \
                        self.__task_prob_int[torch.min(torch.floor(temp_new_state[i, 0]), self.__cpu_max_num).to(torch.int)][
                            torch.min(new_int_num[i], self.__gpu_max_num).to(torch.int)] * 1.0 / self.__task_max_num
                        temp_list[i] = (old_prob - new_prob)
                prob[~indices] = temp_list
                prob[~indices] += (self.__task_prob_int[torch.min(torch.floor(old_state[0]), self.__cpu_max_num).to(torch.int)][1] -
                                   self.__task_prob_float[torch.min(torch.floor(old_state[0]), self.__cpu_max_num).to(torch.int)][
                                       9]) * 1.0 / self.__task_max_num
            old_value = self.__get_value(old_state)
            new_value = self.__get_value(new_state)
            value = old_value - new_value
            value -= 50 * prob
            max_value, max_index = torch.max(value, dim=0)
        return max_value.item(), max_index.item()

    def __get_value(self, state):
        state = state.view(-1, 9)
        expert = get_expert_num(state)
        state = deal_state(state, self._rate)
        value = torch.zeros(len(state), dtype=torch.float32, device=self.__device)
        for i in range(len(self.__trained_experts)):
            indices = expert == i
            temp_state = state[indices]
            if len(temp_state) > 0:
                value[indices] = self.__trained_experts[i](temp_state).reshape(-1)
        return value

#     def run(self, task):
#         self.__buffer.init_again()
#         now_select = -1
#         for i in range(len(self.cluster)):
#             task_cpu, task_gpu = self.get_task_info(task)
#             node = self.cluster[i]
#             temp_node_cpu, temp_node_gpu = self.get_node_info(node)
#             if np.any(task_cpu > temp_node_cpu):
#                 self.__buffer.add_memo(empty=True)
#             else:
#                 state = np.zeros(9)
#                 state[0] = temp_node_cpu.min()
#                 temp_new_state_0 = temp_node_cpu - task_cpu
#                 temp_new_state_0 = temp_new_state_0.min()
#                 temp_node_gpu = temp_node_gpu[:,0].reshape(-1)
#                 temp_gpu = temp_node_gpu[:]
#                 temp_gpu = abs(np.sort(-temp_gpu))
#                 state[1:1+len(temp_gpu)] = temp_gpu
#                 if len(task_gpu) == 0:
#                     next_state = np.ones((8, 9)) * -1
#                     next_state[0,:] = state[:]
#                     next_state[0,0] = temp_new_state_0
#                     state_len = 1
#                     gpu_select = [8]
#                     self.__buffer.add_memo(state, next_state, state_len, gpu_select)
#                     continue
#                 task_gpu = task_gpu[:, 0].sum()
#                 gpu_select = []
#                 state_len = 0
#                 if task_gpu > 1:
#                     next_state = get_next_state(state,np.array([task_cpu.max(),task_gpu]))
#                     state_len = len(next_state)
#                     gpu_select.append(9)
#                 else:
#                     next_state = np.ones((8,9)) * -1
#                     for j in range(len(temp_node_gpu)):
#                         temp_state = np.zeros(8)
#                         if temp_node_gpu[j] >= task_gpu:
#                             next_state[state_len,0] = temp_new_state_0
#                             temp_state[:len(temp_node_gpu)] = temp_node_gpu
#                             temp_state[j] -= task_gpu
#                             gpu_select.append(j)
#                             temp_state = abs(np.sort(-temp_state))
#                             next_state[state_len,1:] = temp_state
#                             state_len += 1
#                 assert state_len == len(gpu_select), "model采样错误"
#                 if state_len == 0:
#                     self.__buffer.add_memo(empty=True)
#                 else:
#                     next_state = np.array(next_state)
#                     gpu_select = np.array(gpu_select)
#                     self.__buffer.add_memo(state,next_state,state_len,gpu_select)
#         state, next_state, state_len, node_select, gpu_select = self.__buffer.sample(self.__device)
#         if len(state) != 0:
#             with torch.no_grad():
#                 state_out = self.__state_value(state)
#                 next_state_out = self.__state_value(next_state)
#                 temp_next_state_out = torch.tensor([torch.min(elem) for elem in torch.split(next_state_out, state_len.tolist())]).to(self.__device)
#                 state_out = state_out.reshape(-1) - temp_next_state_out
#                 max_index = torch.argmax(state_out)
#                 now_select = self.cluster[node_select[max_index]]
#                 elem = torch.split(next_state_out, state_len.tolist())[max_index]
#                 min_index = torch.argmin(elem)
#                 elem = torch.split(gpu_select, state_len.tolist())[min_index]
#                 gpu_select = elem[min_index]
#             if gpu_select == 9:
#                 gpu_select = {}
#                 task_cpu, task_gpu = self.get_task_info(task)
#                 temp_node_cpu, temp_node_gpu = self.get_node_info(now_select)
#                 for i in range(len(task_gpu)):
#                     for j in range(len(temp_node_gpu)):
#                         if np.any(task_gpu[i] > temp_node_gpu[j]):
#                             continue
#                         else:
#                             gpu_select[j] = i
#                             temp_node_gpu[j] -= task_gpu[i]
#                             break
#             elif gpu_select == 8:
#                 gpu_select = {}
#             else:
#                 if gpu_select != 0:
#                     print(gpu_select)
#                 gpu_select = {gpu_select:0}
#
#         if now_select != -1:
#             self.set_task(now_select, task, gpu_select)
#             return True
#         else:
#             return False
#
# class BufferArray:
#     def __init__(self, memo_max_len,state_dim=9):
#         self.__state = np.ones((memo_max_len, state_dim), dtype=np.float32) * -1
#         self.__state_len = np.ones(memo_max_len).astype(int) * -1
#         self.__next_state = np.ones((memo_max_len,state_dim - 1,state_dim), dtype=np.float32) * -1
#         self.__gpu_select = np.zeros((memo_max_len,state_dim - 1)).astype(int) * -1
#         self.__node_select = np.ones(memo_max_len).astype(int) * -1
#         self.__next_idx = 0
#         self.__state_dim = state_dim
#
#     def init_again(self):
#         self.__next_idx = 0
#         self.__state = np.ones_like(self.__state).astype(np.float32) * -1
#         self.__state_len = np.ones_like(self.__state_len).astype(int) * -1
#         self.__next_state = np.ones_like(self.__next_state).astype(np.float32) * -1
#         self.__gpu_select = np.ones_like(self.__gpu_select).astype(int) * -1
#         self.__node_select = np.ones_like(self.__node_select).astype(int) * -1
#
#     def add_memo(self,state=None,next_state=None,state_len=None,gpu_select=None,empty=False):
#         if empty:
#             self.__next_idx += 1
#         else:
#             self.__state[self.__next_idx] = state.reshape(-1,self.__state_dim)
#             self.__state_len[self.__next_idx] = state_len
#             self.__next_state[self.__next_idx] = next_state.reshape(-1, self.__state_dim - 1, self.__state_dim)
#             self.__gpu_select[self.__next_idx,0:len(gpu_select)] = gpu_select
#             self.__node_select[self.__next_idx] = self.__next_idx
#             self.__next_idx += 1
#
#
#     def sample(self, device):
#         indices = self.__node_select >= 0
#         state = self.__state[indices].reshape(-1,self.__state_dim)
#         state_len = self.__state_len[indices]
#         next_state = self.__next_state[indices].reshape(-1,self.__state_dim)
#         gpu_select = self.__gpu_select[indices].reshape(-1)
#         node_select = self.__node_select[indices].reshape(-1)
#         indices = gpu_select >= 0
#         next_state = next_state[indices]
#         gpu_select = gpu_select[indices]
#         state = torch.tensor(state, device=device)
#         next_state = torch.tensor(next_state, device=device)
#         gpu_select = torch.tensor(gpu_select, device=device)
#         return state, next_state, state_len, node_select, gpu_select
