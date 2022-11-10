import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
import random

class NN(nn.Module):
    def __init__(self, inSize, outSize, layers=[], finalActivation=None, activation=torch.tanh,dropout=0.0):
        super(NN, self).__init__()
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
        self.layers.append(nn.Linear(inSize, outSize))
        self.activation = activation
        self.finalActivation = finalActivation
        self.dropout = None
        if dropout > 0:
            self.dropout = torch.nn.Dropout(dropout)

    def setcuda(self, device):
        self.cuda(device=device)

    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = self.activation(x)
            if self.dropout is not None:
                x=self.dropout(x)

            x = self.layers[i](x)

        if self.finalActivation is not None:
            x=self.finalActivation(x)
        return x
    
    def save_model(self,filename):
        torch.save(self,filename)

    def load_model(self,filename):
        return torch.load(filename)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class EGreedyActionSelector(QconAgent):
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, t, **kwargs):
        q_values = self.get(("q_values", t))
        nb_actions = q_values.size()[1]
        size = q_values.size()[0]
        is_random = torch.rand(size).lt(self.epsilon).float()
        random_action = torch.randint(low=0, high=nb_actions, size=(size,))
        max_action = q_values.max(1)[1]
        action = is_random * random_action + (1 - is_random) * max_action
        action = action.long()
        self.set(("action", t), action)