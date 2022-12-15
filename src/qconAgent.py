import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

from utils import NN
import torch


class QconAgent():
    def __init__(self,savedir, nbStep = 10000,batch_size=1):
        self.state_dim = 145
        self.action_dim = 4
        self.save_dir = savedir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.net = NN(145,1).float()
        self.net = self.net.to(device=self.device)

        self.discount_factor = 0.9
        self.learning_rate = 1e-3
        self.nbStep = nbStep
        self.curr_step = 0

        self.saveEvStep = 1000
        self.batch_size = batch_size
        self.memory = deque(maxlen=100000)
        self.sync_every = 1
        self.exploration_rate = 1
        self.learn_every = 1

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def act(self,state):
        if np.random.rand() < self.exploration_rate:
            self.exploration_rate = max(0.1, self.exploration_rate * self.discount_factor)
            action_idx = np.random.randint(self.action_dim)
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()
        self.curr_step += 1
        return action_idx

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def store(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        state = state
        next_state = next_state

        state = torch.tensor(state, device=self.device)
        next_state = torch.tensor(next_state, device=self.device)
        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([done], device=self.device)

        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
    
    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)
