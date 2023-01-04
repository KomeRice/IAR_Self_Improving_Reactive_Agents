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
        sensorDataFourway = [env.mainAgent.getLabeledPositions(orient) for orient in range(4)]
        if doOrientation == -1:
            sensorData = sensorDataFourway[0]
        else:
            sensorData = sensorDataFourway[doOrientation]
        for data in sensorData:
            x, y = data['coords']
            label = data['label']
            o = data['orientation']
            if x < 0 or x > env.cols or y < 0 or y > env.rows or label == 'W':
                continue
            if doOrientation != -1:
                label = data['id']
                sensorNb += 1

            plt.text(x, y, label, va='center', ha='center')
    plt.savefig(filepath)
    plt.close()


class NNDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Linear(input_dim, 512),
            nn.Linear(512, output_dim)]
        )
        self.layers.apply(set_weigths)

    def forward(self, input):
        input = input.type(torch.float32)
        input = self.layers[0](input)
        input = nn.functional.relu(input)
        return self.layers[1](input)


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