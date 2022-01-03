# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
def fix(env, seed):
  env.seed(seed)
  env.action_space.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.set_deterministic(True)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT
import random

# env = gym_tetris.make('TetrisA-v0')
# env = JoypadSpace(env, MOVEMENT)
# fix(env, seed)

# state = env.reset()
# #img = plt.imshow(env.render(mode='rgb_array'))
# done = True
# for step in range(500):
#     if done:
#       state = env.reset()
#     action = env.action_space.sample()
#     state, reward, done, info = env.step(action)
#     env.render()
#     #img.set_data(env.render(mode='rgb_array'))
#     #display.display(plt.gcf())
#     #display.clear_output(wait=True)
# env.close()
class PolicyGradientNetwork(nn.Module):

    def __init__(self, state_dim, aciton_dim):
        super().__init__()
        self.cnn1 = nn.Conv2d(in_channels = 3 , out_channels = 8, kernel_size =3)
        self.max1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels = 8 , out_channels = 8, kernel_size =3)
        self.max2 = nn.MaxPool2d(kernel_size=2)
        self.cnn3 = nn.Conv2d(in_channels = 8 , out_channels = 8, kernel_size =3)
        self.max3 = nn.MaxPool2d(kernel_size=2)
        self.cnn4 = nn.Conv2d(in_channels = 8 , out_channels = 8, kernel_size =3)
        self.max4 = nn.MaxPool2d(kernel_size=2)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(13*14*8, 64)
        self.d1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(64, 32)
        self.d2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(32, 12)
        # self.d3 = nn.Dropout(0.25)
        # self.fc4 = nn.Linear(32, 16)
        # self.d4 = nn.Dropout(0.25)
        # self.fc5 = nn.Linear(16, 12)

    def forward(self, state):
        hid = torch.relu(self.cnn1(state))
        hid = self.max1(hid)
        hid = torch.relu(self.cnn2(hid))
        hid = self.max2(hid)
        hid = torch.relu(self.cnn3(hid))
        hid = self.max3(hid)
        hid = torch.relu(self.cnn4(hid))
        hid = self.max4(hid)
        hid = self.flat(hid)
        
        hid = torch.relu(self.fc1(hid))
        hid = self.d1(hid)
        hid = torch.relu(self.fc2(hid))
        hid = self.d2(hid)
        # hid = torch.relu(self.fc3(hid))
        # hid = self.d3(hid)
        # hid = torch.relu(self.fc4(hid))
        # hid = self.d4(hid)
        
        return F.softmax(self.fc3(hid), dim=-1)
class PolicyGradientAgent():
    
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.model = PolicyGradientNetwork(self.state_dim, self.action_dim)
        self.target_model = PolicyGradientNetwork(self.state_dim, self.action_dim)
        self.target_update()

        self.buffer = ReplayBuffer()
         
    def forward(self, state):
        return self.network(state)
    def learn(self, log_probs, rewards):
        loss = (-log_probs * rewards).sum() # You don't need to revise this to pass simple baseline (but you can)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def sample(self, state):
        state = state[:,:,::-1].copy()
        state = np.transpose(state, (2,0,1))
        state = torch.unsqueeze(torch.FloatTensor(state), dim=0)
        action_prob = network(state)
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob
    def train(self, max_episodes=1000):
        for ep in range(max_episodes):
            done, total_reward = False, 0
            state = self.env.reset()
            while not done:
                action = self.model.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.put(state, action, reward*0.01, next_state, done)
                total_reward += reward
                state = next_state
            if self.buffer.size() >= args.batch_size:
                self.replay()
            self.target_update()
            print('EP{} EpisodeReward={}'.format(ep, total_reward))
            wandb.log({'Reward': total_reward})
    def save(self, PATH): # You should not revise this
        Agent_Dict = {
            "network" : self.network.state_dict(),
            "optimizer" : self.optimizer.state_dict()
        }
        torch.save(Agent_Dict, PATH)

    def load(self, PATH): # You should not revise this
        checkpoint = torch.load(PATH)
        self.network.load_state_dict(checkpoint["network"])
        #如果要儲存過程或是中斷訓練後想繼續可以用喔 ^_^
        self.optimizer.load_state_dict(checkpoint["optimizer"])
from collections import deque
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.995)
parser.add_argument('--eps_min', type=float, default=0.01)

args = parser.parse_args()
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])
    
    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(args.batch_size, -1)
        next_states = np.array(next_states).reshape(args.batch_size, -1)
        return states, actions, rewards, next_states, done
    
    def size(self):
        return len(self.buffer)

import pygame
def move():
    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = 6
                elif event.key == pygame.K_RIGHT:
                    action = 3
                elif event.key == pygame.K_DOWN:
                    action = 9
                elif event.key == pygame.K_UP:
                    action = 1
    return action

network = torch.load("tetris.pt")
# network = PolicyGradientNetwork()
agent = PolicyGradientAgent(network)
agent.network.train()  # 訓練前，先確保 network 處在 training 模式

NUM_BATCH = 100000        # 總共更新 400 次
gamma = 0.99
env = gym_tetris.make('TetrisA-v0')
env = JoypadSpace(env, MOVEMENT)
# fix(env, seed)
prg_bar = (range(NUM_BATCH))
state = env.reset()
times = 0
while times < NUM_BATCH:
    log_probs, rewards = [], []
    
    for i in range(8000):
        action, log_prob = agent.sample(state)
        next_state, reward, done, _ = env.step(action)
        env.render()
        log_probs.append(log_prob)
        rewards.append(reward)
        state = next_state
        if done:
            break
    
    # acculative reward
    length = len(rewards)
    acc = rewards[-1]
    for i in range(1, length):
        rewards[length-1-i]  += acc*gamma
        acc = rewards[length-1-i]
    
    rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)
    agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))   
    torch.save(network, "tetris.pt")
    if done:
        state = env.reset()
        times += 1
env.close()

# 手動
network = PolicyGradientNetwork()
agent = PolicyGradientAgent(network)
agent.network.train()  # 訓練前，先確保 network 處在 training 模式
env = gym_tetris.make('TetrisA-v0')
env = JoypadSpace(env, MOVEMENT)
state = env.reset()
pygame.init()
WIDTH=600
HEIGHT=480
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
env.render()
rewards = []
log_probs = []
done=False
action=9
running=True
while not done:    
    if action !=0:
        state = state[:,:,::-1].copy()
        state = np.transpose(state, (2,0,1))
        state = torch.unsqueeze(torch.FloatTensor(state), dim=0)
        action_prob = network(state)
        action_dist = Categorical(action_prob)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running=False
            break
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                print("left")
                action = 6
            elif event.key == pygame.K_RIGHT:
                print("right")
                action = 3
            elif event.key == pygame.K_DOWN:
                print("down")
                action = 9
            elif event.key == pygame.K_UP:
                print("up")
                action = 1
    if not running : 
        break
    if action !=0:
        action_tensor = torch.from_numpy(np.zeros(1, dtype=int)+int(action))
        state, reward, done, info = env.step(int(action))
        
        log_prob = action_dist.log_prob(action_tensor)
        
        rewards.append(reward)
        log_probs.append(log_prob)
    action=0
    env.render()
env.close()
pygame.quit()
rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)

log_probs = []
for i in range(len(rewards)):
    action, log_prob = agent.sample(state) # at , log(at|st)
    log_probs.append(log_prob)
agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))
torch.save(network, "tetris.pt")
'''
['NOOP' 0,
 'A' 1,
 'B' 2,
 'right' 3,
 'right A' 4,
 'right B' 5,
 'left' 6,
 'left A' 7,
 'left B' 8,
 'down' 9,
 'down A' 10,
 'down B' 11]
'''