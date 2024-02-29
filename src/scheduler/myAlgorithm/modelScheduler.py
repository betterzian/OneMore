from src.scheduler.schedulerClass import Scheduler
import numpy as np
from src.scheduler.myAlgorithm.generateValue.generateContinuousValue import StateValue,get_next_state,BufferArray
import torch
class ModelScheduler(Scheduler):
    def __init__(self,cluster,can_predict = True,task_mem = {},node_mem = {}):
        super().__init__(cluster,can_predict,task_mem,node_mem)
        self.__device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.__state_value = StateValue(9).to(self.__device)
        #self.__state_value.load_state_dict(torch.load('../srcData/offline_task/model/'+ParamHolder().filename+'_model.pth'))
        self.__state_value.load_state_dict(torch.load('../srcData/offline_task/model/off_task_list_model.pth'))
        self.__state_value.eval()
        self.__buffer = BufferArray(len(cluster),1)

    def run(self, task):
        task_cpu, task_gpu = self.get_task_info(task)
        for i in range(len(self.cluster)):
            node = self.cluster[i]
            temp_node_cpu, temp_node_gpu = self.get_node_info(node)
            if np.any(task_cpu > temp_node_cpu):
                self.__buffer.add_memo(empty=True)
            else:
                state = np.zeros(9)
                state[0] = temp_node_cpu.min()
                temp_gpu = temp_node_gpu[:]
                temp_gpu = abs(np.sort(-temp_gpu))
                state[1:len(temp_gpu)] = temp_gpu




    # def run(self, task):
    #     task_cpu, task_gpu = self.get_task_info(task)
    #     task_gpu = task_gpu[:, 0]
    #     now_priority = -1e8
    #     now_select = -1
    #     gpu_select = -1
    #     for node in self.cluster:
    #         temp_select = {}
    #         temp_node_cpu, temp_node_gpu = self.get_node_info(node)
    #         temp_node_gpu = temp_node_gpu[:, 0]
    #         if np.any(task_cpu > temp_node_cpu):
    #             continue
    #         else:
    #             old_state = np.zeros(9)
    #             new_state = np.zeros(9)
    #             old_state[0] = temp_node_cpu.min()
    #             new_state[0] = temp_node_cpu.min() - task_cpu.max()
    #             temp_new_node_gpu = temp_node_gpu[:]
    #             temp_new_node_gpu = abs(np.sort(-temp_new_node_gpu))
    #             old_state[1:1+len(temp_new_node_gpu)] = temp_new_node_gpu
    #             old_value = self.__state_value(torch.tensor(old_state,dtype=torch.float32).to(self.__device))
    #             task_gpu = task_gpu.sum()
    #             if task_gpu < 1:
    #                 temp_gpu_select = -1
    #                 for i in range(len(temp_node_gpu)):
    #                     temp_new_node_gpu = temp_node_gpu[:]
    #                     if temp_node_gpu[i] > task_gpu:
    #                         temp_new_node_gpu[i] -= task_gpu
    #                         temp_new_node_gpu = abs(np.sort(-temp_new_node_gpu))
    #                         new_state[1:1+len(temp_new_node_gpu)] = temp_new_node_gpu
    #                         new_value = self.__state_value(torch.tensor(new_state,dtype=torch.float32).to(self.__device))
    #                         if now_priority < old_value - new_value:
    #                             temp_gpu_select = i
    #                 if temp_gpu_select != -1:
    #                     temp_select[temp_gpu_select] = 0
    #                     gpu_select = temp_select
    #                     now_select = node
    #             else:
    #                 j = 0
    #                 temp_task_gpu = task_gpu
    #                 for i in range(len(temp_node_gpu)):
    #                     if temp_task_gpu > 0 and temp_node_gpu[i] == 1:
    #                         temp_task_gpu -= 1
    #                         temp_node_gpu[i] -= 1
    #                         temp_select[i] = j
    #                         j += 1
    #                 if len(temp_select) == int(task_gpu):
    #                     temp_node_gpu = abs(np.sort(-temp_node_gpu))
    #                     new_state[1:1+len(temp_node_gpu)] = temp_node_gpu
    #                     new_value = self.__state_value(torch.tensor(new_state, dtype=torch.float32).to(self.__device))
    #                     if now_priority < old_value - new_value:
    #                         gpu_select = temp_select
    #                         now_select = node
    #     if now_select != -1:
    #         self.set_task(now_select, task, gpu_select)
    #         return True
    #     else:
    #         return False
