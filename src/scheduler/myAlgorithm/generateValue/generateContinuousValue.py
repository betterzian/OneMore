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


if __name__ == "__main__":
    state_value = StateValue(9)
    epoch = 1000
    task_list = []
    for _ in range(epoch):
        fail_num = 0
        node_info = np.random.rand(1, 9)
        node_info = np.round(node_info, decimals=1)

        for _ in range(10):
            task = task_list[random.randint(0, len(task_list))]
            task_cpu, task_gpu = self.get_task_info(task)
            if node_cpu >= task_cpu and node_gpu >= task_gpu:
                node_cpu -= task_cpu
                node_gpu -= task_gpu
            else:
                fail_num += 1
        data = np.concatenate((node_cpu, node_gpu, node_cpu.sum() + node_gpu.sum() * ParamHolder().cpu_gpu_rate),axis=1)
        data = pd.DataFrame(data)
        data.to_csv('../data_src/offline_task/off_task_list_data' + '.txt', index=False, header=True, sep='\t',
                    mode='a', encoding="utf-8", quoting=csv.QUOTE_NONE, escapechar=',')
