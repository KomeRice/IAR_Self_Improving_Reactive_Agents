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
            nn.Linear(30, outSize),
            activation
        )
        self.target = copy.deepcopy(self.online)

    def setcuda(self, device):
        self.cuda(device=device)

    def forward(self, input, model):
        input = input.type(torch.float32)
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def save_model(self, filename):
        torch.save(self, filename)

    def load_model(self, filename):
        return torch.load(filename)


def plot_examples(data):
    cmap = ListedColormap(["white", "black", "red", "blue", "green"])
    plt.imshow(np.array(data), cmap=cmap)
    plt.show()


def save_plot(data, filepath):
    cmap = ListedColormap(["white", "black", "red", "blue", "green"])
    plt.imshow(np.array(data), cmap=cmap)
    plt.savefig(filepath)


class NNDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.online = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        input = input.type(torch.float32)
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)


class Activation_Sigmoid(nn.Module):
    def __init__(self):
        super(Activation_Sigmoid, self).__init__()

    @torch.no_grad()
    def forward(self, inputs):
        # Save input and calculate/save output
        # of the sigmoid function
        return 2 * (torch.sigmoid(inputs) - 0.5)
