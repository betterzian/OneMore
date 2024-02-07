import torch.nn as nn
import torch.nn.functional as F
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
    pass