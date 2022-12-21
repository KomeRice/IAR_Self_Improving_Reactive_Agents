import random
import numpy as np
from collections import deque
from utils import NN
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

        # discount factor
        self.gamma = 0.9
        self.exploration_rate = 1
        self.Temperature = 0.05
        self.learning_rate = 1e-3

        # max step
        self.nbStep = nbStep
        self.curr_step = 0

        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.sync_every = 1
        self.learn_every = 1

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.env = env

    def act(self, state):
        """Predict and return the best action given the state
        :param state: list[int] observation of the state
        :param test: bool check if training or testing mode
        :return: action_idx : int the index of the best action given the state
        """
        if not self.test and np.random.rand() < self.exploration_rate:
            self.exploration_rate = max(0.1, self.exploration_rate - self.Temperature)
            action_idx = np.random.randint(self.action_dim)
        else:
            Q = np.zeros(4)
            for a in range(4):
                state = torch.FloatTensor(state)
                Q[a] = self.net(state, model="online")
                state = self.env.mainAgent.observation(a)
            action_idx = np.argmax(Q)
        self.curr_step += 1
        return action_idx

    def td_estimate(self, state, action):
        """ Compute the Q values of the given state
        :param action: action taken
        :param state: current state
        :return: Q values
        """
        Q = torch.zeros((self.batch_size, self.action_dim)).to(self.device)
        for a in range(self.action_dim):
            state = torch.FloatTensor(state)
            Q[:, a] = self.net(state, model="online").view(-1)
            state = self.env.mainAgent.observation(a)
        return Q[:, int(action.item())]

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        """ Aggregate current reward and all the estimated next rewards
        :param reward: current reward
        :param next_state: next possible state
        :param done: done
        :return: y the aggregation of the rewards
        """
        next_Q = torch.zeros((self.batch_size, self.action_dim)).to(self.device)
        for a in range(self.action_dim):
            next_state = torch.FloatTensor(next_state)
            next_Q[:, a] = self.net(next_state, model="target").view(-1)
            next_state = self.env.mainAgent.observation(a)
        return reward + (1 - done) * self.gamma * torch.max(next_Q, dim=1)[0]

    def update_Q_online(self, td_estimate, td_target):
        """Update the learning network givent the Q values and target action
        :param td_estimate: Q values
        :param td_target: aggregation of rewards
        :return:
        """
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss = torch.autograd.Variable(loss, requires_grad=True)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        """
        Sync the online and target network
        """
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

        state = torch.FloatTensor(state, device=self.device)
        next_state = torch.FloatTensor(next_state, device=self.device)
        action = torch.FloatTensor([action], device=self.device)
        reward = torch.FloatTensor([reward], device=self.device)
        done = torch.FloatTensor([done], device=self.device)

        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def learn(self):
        """train the network
        :return: mean of Q values,loss
        """
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

        return td_est.mean().item(), loss

    def save(self, outputDir):
        """Save model to output directory
        """
        torch.save(self.net.state_dict(), outputDir)

    def load(self, inputDir):
        """Load model from input directory
        """
        self.net.load_state_dict(torch.load(inputDir, map_location=self.device))
