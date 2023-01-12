import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
import gameAgents as ag

import matplotlib.pyplot as plt
import copy
from matplotlib.colors import ListedColormap

obs_rotator = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,0,1,2,3,4,
                       23,24,25,26,27,28,29,30,31,20,21,22,
                       34,35,36,37,38,39,32,33,
                       42,43,44,45,46,47,40,41,
                       49,50,51,48,
                       55,56,57,58,59,60,61,62,63,52,53,54,
                       66,67,68,69,70,71,64,65,
                       74,75,76,77,78,79,72,73,
                       81,82,83,80,
                       88,89,90,91,92,93,94,95,96,97,98,99,84,85,86,87,
                       103,104,105,106,107,108,109,110,111,100,101,102,
                       114,115,116,117,118,119,112,113,
                       121,122,123,120,
                       124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,
                       143,140,141,142,
                       144]


class NN(nn.Module):
    def __init__(self, inSize, outSize):
        super(NN, self).__init__()
        activation = Activation_Sigmoid()
        self.online = nn.Sequential(
            nn.Linear(inSize, 30),
            activation,
            nn.Linear(30, outSize)
        )
        self.online.apply(set_weigths)

    def setcuda(self, device):
        self.cuda(device=device)

    def forward(self, input):
        input = input.type(torch.float32)
        return self.online(input)

    def save_model(self, filename):
        torch.save(self, filename)

    def load_model(self, filename):
        return torch.load(filename)


def plot_examples(data):
    cmap = ListedColormap(["white", "black", "red", "blue", "green"])
    plt.imshow(np.array(data), cmap=cmap)
    plt.show()

def save_plot(data, filepath, env=None, showSensors=False, doOrientation=-1):
    plt.figure()
    cmap = ListedColormap(["white", "black", "red", "blue", "green"])
    plt.imshow(np.array(data), cmap=cmap)
    sensorNb = 0
    if showSensors:
        sensorDataFourway = [env.mainAgent.getLabeledPositions() for orient in range(4)]
        if doOrientation == -1:
            sensorData = sensorDataFourway[0]
        else:
            sensorData = sensorDataFourway[doOrientation]
        for data in sensorData:
            x, y = data['coords']
            label = data['label']
            o = 0
            if x < 0 or x > env.cols or y < 0 or y > env.rows or label == 'W':
                continue
            if doOrientation != -1:
                label = data['id']
                sensorNb += 1

            try:
                if env.isAgentAt(x, y):
                    agent = env.getAgentAt(x, y)
                    if isinstance(agent, ag.FoodAgent):
                        plt.text(x, y, label, va='center', ha='center', color='red')
                    elif isinstance(agent, ag.EnemyAgent):
                        plt.text(x, y, label, va='center', ha='center', color='blue')
                elif env.at(x, y) == 'O':
                    plt.text(x, y, label, va='center', ha='center', color='yellow')
                else:
                    plt.text(x, y, label, va='center', ha='center')
            except IndexError:
                plt.text(x, y, label, va='center', ha='center')

    plt.savefig(filepath)
    plt.close()


class NNDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim,512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self,x):
        return self.layers(x)


class Activation_Sigmoid(nn.Module):
    def __init__(self):
        super(Activation_Sigmoid, self).__init__()

    @torch.no_grad()
    def forward(self, inputs):
        # Save input and calculate/save output
        # of the sigmoid function
        return 2 * (torch.sigmoid(inputs) - 0.5)


def set_weigths(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # get the number of the inputs
        m.weight.data.uniform_(-0.1, 0.1)
        m.bias.data.fill_(0)

def obs_rot90(obs):
    return np.array(obs)[obs_rotator]

def get_rotator():
    return obs_rotator