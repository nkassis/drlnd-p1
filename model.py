import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """ 
    Implements a Deep Q Network with 3 fully connected layers
    """

    def __init__(self, state_size, action_size, fc1_units=128, fc2_units=128, seed=None):
        super(QNetwork, self).__init__()

        if seed:
            torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
