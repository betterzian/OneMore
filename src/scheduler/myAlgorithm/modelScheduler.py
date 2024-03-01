from src.scheduler.schedulerClass import Scheduler
import numpy as np
from src.scheduler.myAlgorithm.generateValue.generateContinuousValue import StateValue,get_next_state
import torch
class ModelScheduler(Scheduler):
    def __init__(self,cluster,can_predict = True,task_mem = {},node_mem = {}):
        super().__init__(cluster,can_predict,task_mem,node_mem)
        self.__device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.__state_value = StateValue(9).to(self.__device)
        #self.__state_value.load_state_dict(torch.load('../srcData/offline_task/model/'+ParamHolder().filename+'_model.pth'))
        self.__state_value.load_state_dict(torch.load('../srcData/offline_task/model/off_task_list_model.pth'))
        self.__state_value.eval()
        # self.__buffer = BufferArray(len(cluster))

    def run(self, task):
        task_cpu, task_gpu = self.get_task_info(task)
        if len(task_gpu) == 0:
            task_gpu = np.array([0])
        else:
            task_gpu = task_gpu[:, 0]
        now_priority = -1e8
        now_select = -1
        gpu_select = {}
        for node in self.cluster:
            temp_select = {}
            temp_node_cpu, temp_node_gpu = self.get_node_info(node)
            temp_node_gpu = temp_node_gpu[:, 0]
            if np.any(task_cpu > temp_node_cpu):
                continue
            else:
                old_state = np.zeros(9)
                old_state[0] = temp_node_cpu.min()
                temp_new_state_0 = temp_node_cpu - task_cpu
                temp_new_state_0 = temp_new_state_0.min()
                temp_new_node_gpu = temp_node_gpu[:]
                temp_new_node_gpu = abs(np.sort(-temp_new_node_gpu))
                old_state[1:1+len(temp_new_node_gpu)] = temp_new_node_gpu
                old_value = self.__state_value(torch.tensor(old_state,dtype=torch.float32).to(self.__device))
                task_gpu = task_gpu.sum()
                if task_gpu == 0:
                    new_state = old_state - 0
                    new_state[0] = temp_new_state_0
                    new_value = self.__state_value(torch.tensor(new_state,dtype=torch.float32).to(self.__device))
                    if now_priority < old_value - new_value:
                        gpu_select = {}
                        now_select = node
                elif task_gpu < 1:
                    gpu_list = []
                    gpu_num = []
                    for i in range(len(temp_node_gpu)):
                        temp_new_node_gpu = temp_node_gpu.copy()
                        new_state = np.zeros(9)
                        if temp_node_gpu[i] > task_gpu:
                            temp_new_node_gpu[i] -= task_gpu
                            temp_new_node_gpu = abs(np.sort(-temp_new_node_gpu))
                            new_state[0] = temp_new_state_0
                            new_state[1:1+len(temp_new_node_gpu)] = temp_new_node_gpu
                            new_state = torch.tensor(new_state,dtype=torch.float32).to(self.__device)
                            gpu_list.append(new_state)
                            gpu_num.append(i)
                    if len(gpu_list) > 0:
                        new_state = gpu_list[0].reshape(-1,9)
                        for i in range(1,len(gpu_list)):
                            new_state = torch.cat((new_state, gpu_list[i].reshape(-1,9)), dim=0)
                        new_value = self.__state_value(new_state)
                        if now_priority < old_value - new_value.min():
                            gpu_select = {gpu_num[torch.argmin(new_value)]:0}
                            now_select = node
                else:
                    j = 0
                    new_state = np.zeros(9)
                    temp_task_gpu = task_gpu
                    for i in range(len(temp_node_gpu)):
                        if temp_task_gpu > 0 and temp_node_gpu[i] == 1:
                            temp_task_gpu -= 1
                            temp_node_gpu[i] -= 1
                            temp_select[i] = j
                            j += 1
                    if len(temp_select) == int(task_gpu):
                        temp_node_gpu = abs(np.sort(-temp_node_gpu))
                        new_state[1:1+len(temp_node_gpu)] = temp_node_gpu
                        new_value = self.__state_value(torch.tensor(new_state, dtype=torch.float32).to(self.__device))
                        if now_priority < old_value - new_value:
                            gpu_select = temp_select
                            now_select = node
        if now_select != -1:
            self.set_task(now_select, task, gpu_select)
            return True
        else:
            return False


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