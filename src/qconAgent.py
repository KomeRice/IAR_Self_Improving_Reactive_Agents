import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

from utils import NN
import torch

#use gpu if not then cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class QconAgent(NN):
    def __init__(self, nbStep = 10000,batch_size=32):
        self.inNN = NN(28,1,activation=torch.nn.ReLU()).to(device)
        self.targetNN = NN(28,1,activation=torch.nn.ReLU()).to(device)
        self.targetNN.load_state_dict(self.net.state_dict())
        self.epsilon = 1
        self.discount_factor = 0.9
        self.learning_rate = 1e-3
        self.nbStep = nbStep
        self.saveEvStep = 1000
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=self.lr)


    def train(self,nbStep):
        raise NotImplementedError

    
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