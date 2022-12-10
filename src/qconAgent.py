import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

from utils import NN
import torch


class QconAgent(NN):
    def __init__(self,savedir, nbStep = 10000,batch_size=32):
        self.state_dim = 145
        self.action_dim = 4
        self.save_dir = savedir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.net = NN(145,1).float()
        self.net = self.net.to(device=self.device)

        self.epsilon = 1
        self.discount_factor = 0.9
        self.learning_rate = 1e-3
        self.nbStep = nbStep
        self.currStep = 0

        self.saveEvStep = 1000
        self.batch_size = batch_size
        self.memory = deque(maxlen=100000)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()


    def train(self,batch):
        self.optimizer.zero_grad()
        batch_size = len(batch)
        ob, action, reward, new_ob, done = map(list,zip(*batch))
        ob_v=torch.tensor(np.array(ob).reshape(-1,145),dtype=torch.float32,device=Device)
        action_v=torch.tensor(np.array(action).reshape(-1),dtype=torch.float32,device=Device)
        new_ob_v=torch.tensor(np.array(new_ob).reshape(-1,145),dtype=torch.float32,device=Device)
        reward_v=torch.tensor(np.array(reward).reshape(-1),dtype=torch.float32,device=Device)
        done_v=torch.tensor(np.array(done).reshape(-1),dtype=torch.float32,device=Device)
        Q_next=torch.zeros((self.batch_size,4)).to(Device)
        for a in range(4):
            Q_next[:,a]=self.target_net(new_ob_v).view(-1)
            new_ob_v=new_ob_v[[[i]*145 for i in range(self.batch_size)],[self.rotation for _ in range(self.batch_size)]]
        y=reward_v+(1-done_v)*self.gamma*torch.max(Q_next,dim=1)[0]
        Q=torch.zeros((self.batch_size,4)).to(Device)
        for a in range(4):
            Q[:,a]=self.net(ob_v).view(-1)
            ob_v=ob_v[[[i]*145 for i in range(self.batch_size)],[self.rotation for _ in range(self.batch_size)]]
        loss=torch.nn.L1Loss()(y.detach(),Q[range(batch_size),np.array(action_v.detach().cpu().numpy())])
        loss.backward()
        self.optimizer.step()

    def predict(self,obs):
        action = [self.moveUp, self.moveDown, self.moveLeft, self.moveRight]
        obs = np.array(obs)
        self.epsilon = max(0.1,self.epsilon-1/150000)
        self.nb_iter +=1
        if random.random()<self.epsilon:
            return random.choice(action)
        else:
            Q = np.zeros(4)
            for a in range(4):
                Q[a]=self.net(torch.tensor([obs],dtype=torch.float32)).detach().cpu().numpy()[0]
                obs = self.doAllSensor(a) #TODO change this after
            return np.argmax(Q)

    def act(self,state):
    
        if np.random.rand() < self.exploration_rate:
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

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

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
