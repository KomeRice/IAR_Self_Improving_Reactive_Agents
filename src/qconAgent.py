import copy
import random
import numpy as np
from collections import deque
from utils import NN, NNDQN, obs_rot90
import torch


class QconAgent:
    def __init__(self, savedir, env=None, nbStep=10000, batch_size=1, memory_size=1, test=False):
        self.test = test
        # input dim and action dim
        self.state_dim = 145
        self.action_dim = 4
        self.save_dir = savedir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 145 input neural network
        self.net = NN(145, 1).float()
        self.net = self.net.to(device=self.device)
        self.target_net = copy.deepcopy(self.net)
        # Q_target parameters are frozen.
        for p in self.target_net.parameters():
            p.requires_grad = False

        # discount factor
        self.gamma = 0.9
        self.Temperature = 0.05
        self.learning_rate = 0.3

        # max step
        self.nbStep = nbStep
        self.curr_step = 0

        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.sync_every = 1

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.loss_fn = torch.nn.L1Loss()

        self.env = env

    def act(self, state):
        """Predict and return the best action given the state
        :param state: list[int] observation of the state
        :return: action_idx : int the index of the best action given the state
        """
        t = self.Temperature
        Q = np.zeros(4)
        self.net.eval()
        with torch.no_grad():
            for a in range(4):
                state = torch.FloatTensor(state)
                Q[a] = self.net(state).view(-1)
                state = torch.tensor(obs_rot90(state))
        self.net.train()
        if self.test:
            action_idx = np.argmax(Q)
        else:
            prob = []
            for a in range(4):
                prob.append(np.exp(Q[a] / t))
            total = np.sum(prob)
            p = [i / total for i in prob]
            action_idx = np.random.choice(range(4), p=p)
        self.curr_step += 1
        return action_idx

    def sync_Q_target(self):
        """
        Sync the online and target network
        """
        self.target_net.load_state_dict(self.net.state_dict())

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

        state = torch.FloatTensor(state, device=self.device)
        next_state = torch.FloatTensor(next_state, device=self.device)
        action = torch.IntTensor([action], device=self.device)
        reward = torch.FloatTensor([reward], device=self.device)
        done = torch.FloatTensor([done], device=self.device)

        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        to_sample = min(len(self.memory), self.batch_size)
        batch = []
        for i in range(to_sample):
            n = len(self.memory)
            w = min(3.0, 1 + 0.02 * n)
            r = random.uniform(0, 1)
            k = n * np.log(1 + r * (np.exp(w) - 1)) / w
            batch.append(self.memory[int(k)])
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def learn(self,course):
        """train the network
        :return: mean of Q values,loss
        """

        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        # Sample from memory
        state, next_state, action, reward, done = course

        # Get TD Estimate and TD Target
        Q = torch.zeros(4).to(self.device)
        next_Q = torch.zeros(4).to(self.device)
        self.net.train()
        for a in range(4):
            Q[a] = self.net(state).view(-1)
            state = torch.tensor(obs_rot90(state.squeeze()))
        self.net.eval()
        for a in range(4):
            next_Q[a] = self.target_net(next_state).view(-1)
            next_state = torch.tensor(obs_rot90(state.squeeze()))
        td_tgt = reward + (1 - done) * self.gamma * torch.max(next_Q)
        # Backpropagate loss
        self.net.train()
        loss = self.loss_fn(td_tgt,Q[action.item()])
        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()

        return

    def batchlearn(self):
        if len(self.memory) < self.batch_size:
            return

        st,nx_st,ac,reward,done = self.recall()
        if self.batch_size ==1:
            self.learn((st,nx_st,ac,reward,done))
        else:
            for course in zip(st,nx_st,ac,reward,done):
                self.learn(course)
    def save(self, outputDir):
        """Save model to output directory
        """
        torch.save(self.net.state_dict(), outputDir)

    def load(self, inputDir):
        """Load model from input directory
        """
        self.net.load_state_dict(torch.load(inputDir, map_location=self.device))


class DQNAgent(QconAgent):
    def __init__(self, savedir, env=None, nbStep=10000, batch_size=1, memory_size=100, test=False):
        super().__init__(savedir, env, nbStep, batch_size, memory_size, test)
        self.test = False

        self.net = NNDQN(145, 1).float()
        self.net = self.net.to(device=self.device)
        self.target_net = copy.deepcopy(self.net)
        # Q_target parameters are frozen.
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.learning_rate = 1e-3
        self.sync_every = 1e4
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.batch_size = 32

        self.EPS_START = 0.9
        self.EPS_END = 0.1
        self.EPS_DECAY = 1e-7
        self.TAU = 0.005

        self.loss_fn = torch.nn.MSELoss()

    def act(self, state):
        """Predict and return the best action given the state using E greedy
        :param state: list[int] observation of the state
        :return: action_idx : int the index of the best action given the state
        """
        Q = np.zeros(4)
        self.net.eval()
        with torch.no_grad():
            for a in range(4):
                state = torch.FloatTensor(state)
                Q[a] = self.net(state).view(-1)
                state = torch.tensor(obs_rot90(state))
        self.net.train()
        if self.test:
            action_idx = np.argmax(Q)
        else:
            p = random.random()
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * np.exp(
                -1. * self.curr_step / self.EPS_DECAY)
            if p > eps_threshold:
                action_idx = np.argmax(Q)
            else:
                action_idx = random.randint(0,3)
        self.curr_step += 1
        return action_idx
