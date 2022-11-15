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
        self.inNN = NN(145,1,activation=torch.nn.ReLU()).to(device)
        self.targetNN = NN(145,1,activation=torch.nn.ReLU()).to(device)
        self.targetNN.load_state_dict(self.net.state_dict())
        self.epsilon = 1
        self.discount_factor = 0.9
        self.learning_rate = 1e-3
        self.nbStep = nbStep
        self.saveEvStep = 1000
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=self.lr)


    def train(self):
        self.optimizer.zero_grad()
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
        loss=torch.nn.MSELoss()(y.detach(),Q[range(self.batch_size),np.array(action_v.detach().cpu().numpy())])
        loss.backward()
        self.optimizer.step()
        if self.nb_iter%self.C==0:
            self.target_net.load_state_dict(self.net.state_dict())  

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