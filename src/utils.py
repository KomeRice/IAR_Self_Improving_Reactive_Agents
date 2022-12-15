import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque

import matplotlib.pyplot as plt
import copy
from matplotlib.colors import ListedColormap

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
        
def plot_examples(data):
    cmap = ListedColormap(["white", "black", "red", "blue", "green"])
    plt.imshow(np.array(data),cmap=cmap)
    plt.show()