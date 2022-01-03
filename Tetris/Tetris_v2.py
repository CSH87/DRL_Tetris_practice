import numpy as np
import gym_tetris
import argparse
import random
from collections import deque
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import MOVEMENT
from skimage import color
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--loss_function', type=str, default="mse")
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--eps', type=float, default=1.0) # the exploration probability
parser.add_argument('--eps_decay', type=float, default=0.999) # 1000 episodes to 0.367 eps
parser.add_argument('--eps_min', type=float, default=0.01)
parser.add_argument('--state_dim', type=tuple, default=(240, 256, 1))
parser.add_argument('--frameskip', type=int, default=4) # skip the frame
parser.add_argument('--update', type=int, default=10) # x episode update target model
parser.add_argument('--model_path', type=str, default="tetris.ckpt") # store the model
parser.add_argument('--batch_size', type=int, default=32) # replay batch_size
parser.add_argument('--replay', type=int, default=32) # replay how many batch
args = parser.parse_args()

class ReplayBuffer:
    def __init__(self, capacity=10240):
        self.buffer = deque(maxlen=capacity)
        self.shape = (args.batch_size, ) + args.state_dim
    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])
    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done =                                                           map(np.asarray, zip(*sample))
        states = np.array(states).reshape(self.shape)
        next_states = np.array(next_states).reshape(self.shape)
        return states, actions, rewards, next_states, done
    def size(self):
        return len(self.buffer)

class ActionStateModel:
    def __init__(self, state_dim, aciton_dim):
        self.state_dim  = state_dim
        self.action_dim = aciton_dim
        self.epsilon = args.eps
        self.model = self.build_model()
    def build_model(self):
        inputs = Input(self.state_dim,)
        output = Conv2D(12, kernel_size=3, activation="relu")(inputs)
        output = MaxPooling2D(pool_size=(2, 2))(output)
        output = Conv2D(12, kernel_size=3, activation="relu")(output)
        output = MaxPooling2D(pool_size=(2, 2))(output)
        output = Conv2D(12, kernel_size=3, activation="relu")(output)
        output = MaxPooling2D(pool_size=(2, 2))(output)
        output = Flatten()(output)
        
        output = Dense(64, activation='relu')(output)
        output = Dense(32, activation='relu')(output)
        output = Dense(16, activation='relu')(output)
        output = Dense(self.action_dim)(output)
        model = Model(inputs, output)
        model.compile(loss=args.loss_function, optimizer=Adam(args.lr))
        return model
    def predict(self, state):
        return self.model.predict(state)
    def get_action(self, state):
        state = np.expand_dims(state, axis=0)
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)
        else:
            q_value = self.predict(state)[0]
            return np.argmax(q_value)
    def epsilon_decay(self):
        if self.epsilon > args.eps_min:
            self.epsilon *= args.eps_decay
        else:
            self.epsilon = args.eps_min
    def train(self, states, targets):
        self.model.fit(states, targets, epochs=1, verbose=0)
    def summary(self):
        return self.model.summary()
    def save(self, path=args.model_path):
        self.model.save(path)
        with open("epsilon.txt", "w") as f:
            f.write(str(self.epsilon))
    def load(self, path=args.model_path):
        self.model = load_model(path)
        with open("epsilon.txt", "r") as f:
            eps = f.read()
        self.epsilon = float(eps)
        print("epsilon:{}".format(eps))
        
class Agent:
    def __init__(self, env):
        self.env = env
        self.current_episode = 1
        self.state_dim = args.state_dim
        self.frameskip = args.frameskip
        self.action_dim = self.env.action_space.n
        self.model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_update()
        self.buffer = ReplayBuffer()
    def target_update(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)
    def replay(self):
        for _ in range(args.replay):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.target_model.predict(states)
            next_q_values = self.target_model.predict(next_states).max(axis=1)
            targets[range(args.batch_size), actions] = rewards + (1-done) * next_q_values * args.gamma
            self.model.train(states, targets)
    def to_gray(self, state):
        return color.rgb2gray(state).reshape(args.state_dim)
    def train(self, episodes=1000, show=False):
        # dict_action = {0:"NOOP", 1:"A", 2:"B", 3:"right", 4:"left", 5:"down"}
        for episode in range(self.current_episode, self.current_episode + episodes):
            print('EP{} start'.format(episode))
            done, total_reward = False, 0
            state = self.env.reset()
            state = self.to_gray(state)
            frame = 0
            while not done:
                if frame%self.frameskip==0:    
                    frame = 0
                    action = self.model.get_action(state)
                # print(dict_action[action])
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.to_gray(next_state)
                if show:
                    self.env.render()
                self.buffer.put(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
                frame += 1
            if self.buffer.size() >= args.batch_size:
                self.replay()
            self.model.epsilon_decay()
            if episode%args.update==0:
                self.target_update()
                self.current_episode = episode
                self.save()
            print('EP{} : Reward={}\n'.format(episode, total_reward))
        self.env.close()
    def save(self):
        self.model.save()
        with open("episode.txt", "w") as f:
            f.write(str(self.current_episode))
    def load(self):
        self.model.load()
        self.target_model.load()
        with open("episode.txt", "r") as f:
            episode = f.read()
        self.current_episode = int(episode)+1

def main():
    env = gym_tetris.make('TetrisA-v3')
    env = JoypadSpace(env, MOVEMENT)
    agent = Agent(env)
    agent.load()
    agent.train(episodes=10000, show=True)
    
if __name__ == "__main__":
    main()
