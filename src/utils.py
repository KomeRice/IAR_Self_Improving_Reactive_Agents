import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import math

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

    def chooseToSave(self,m):#appendix b
        w = min(3,1+0.02*m)
        r = random.random()
        return int(m*math.log(1+r*(math.e**w-1))/w)

