import copy
import random
import numpy as np
from collections import deque
from utils import NN, NNDQN
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
        if self.test:
            for a in range(4):
                Q[a] = self.net(torch.FloatTensor(state[a]))
            action_idx = np.argmax(Q)
        else:
            prob = []
            for a in range(4):
                Q[a] = self.net(torch.FloatTensor(state[a]))
                prob.append(np.exp(Q[a] / t))
            total = np.sum(prob)
            p = [i / total for i in prob]
            action_idx = np.random.choice(range(4), p=p)
        self.curr_step += 1
        return action_idx

    def td_estimate(self, state, action):
        """ Compute the Q values of the given state
        :param action: action taken
        :param state: current state
        :return: Q values
        """
        to_sample = min(len(self.memory), self.batch_size)
        Q = torch.zeros((to_sample, self.action_dim)).to(self.device)
        for a in range(self.action_dim):
            Q[:, a] = self.net(state[:, a]).view(-1)
        return Q[range(to_sample), np.array(action.detach().cpu().numpy())]

    def td_target(self, reward, next_state, done):
        """ Aggregate current reward and all the estimated next rewards
        :param reward: current reward
        :param next_state: next possible state
        :param done: done
        :return: y the aggregation of the rewards
        """
        to_sample = min(len(self.memory), self.batch_size)
        next_Q = torch.zeros((to_sample, self.action_dim)).to(self.device)
        for a in range(self.action_dim):
            next_Q[:, a] = self.target_net(next_state[:, a]).view(-1)
        return reward + (1 - done) * self.gamma * torch.max(next_Q, dim=1)[0]

    def update_Q_online(self, td_estimate, td_target):
        """Update the learning network givent the Q values and target action
        :param td_estimate: Q values
        :param td_target: aggregation of rewards
        :return:
        """
        self.optimizer.zero_grad()
        loss = self.loss_fn(td_estimate, td_target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

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

    def learn(self):
        """train the network
        :return: mean of Q values,loss
        """
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        # td_est = self.td_estimate(state, action)
        to_sample = min(len(self.memory), self.batch_size)

        # Get TD Target
        # td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        # loss = self.update_Q_online(td_est, td_tgt.detach())

        Q = torch.zeros((to_sample, self.action_dim)).to(self.device)
        next_Q = torch.zeros((to_sample, self.action_dim)).to(self.device)
        for a in range(4):
            Q[:, a] = self.net(state[:, a]).view(-1)
            next_Q[:, a] = self.target_net(next_state[:, a]).view(-1)
        td_est = Q[range(to_sample), np.array(action.detach().cpu().numpy())]
        td_tgt = reward + (1 - done) * self.gamma * torch.max(next_Q, dim=1)[0]

        loss = self.loss_fn(td_est, td_tgt)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return td_est.mean().item(), loss.item()

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

    def act(self, state):
        """Predict and return the best action given the state using E greedy
        :param state: list[int] observation of the state
        :return: action_idx : int the index of the best action given the state
        """
        Q = np.zeros(4)
        if self.test:
            for a in range(4):
                Q[a] = self.net(torch.FloatTensor(state[a]))
            action_idx = np.argmax(Q)
        else:
            p = random.random()
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * np.exp(
                -1. * self.curr_step / self.EPS_DECAY)
            if p > eps_threshold:
                with torch.no_grad():
                    for a in range(4):
                        Q[a] = self.net(torch.FloatTensor(state[a]))
                    action_idx = np.argmax(Q)
            else:
                action_idx = random.randint(0,3)
        self.curr_step += 1
        return action_idx
