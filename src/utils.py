import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import math
import copy

class NN(nn.Module):
    def __init__(self, inSize, outSize):
        super(NN, self).__init__()
        self.online = nn.Sequential(
            nn.Linear(inSize,30),
            nn.Sigmoid()*2-1,
            nn.Linear(30,outSize),
            nn.Sigmoid()*2-1
        )
        self.target = copy.deepcopy(self.online)

    def setcuda(self, device):
        self.cuda(device=device)

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
    
    def save_model(self,filename):
        torch.save(self,filename)

    def load_model(self,filename):
        return torch.load(filename)
    
    def loss(self,previous_action,u,Ui):
        """previous action example = [0,0,1,0] (go to east)"""
        return previous_action*(u-Ui)
        


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

