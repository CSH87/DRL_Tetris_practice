import gym_tetris
import argparse
import random
import numpy as np
from collections import deque
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import MOVEMENT
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--loss_function', type=str, default="mse")
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.9995)
parser.add_argument('--eps_min', type=float, default=0.01)
args = parser.parse_args()

class ReplayBuffer:
    def __init__(self, capacity=10240):
        self.buffer = deque(maxlen=capacity)
    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])
    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done =                                                           map(np.asarray, zip(*sample))
        states = np.array(states).reshape(args.batch_size, 240, 256, 3)
        next_states = np.array(next_states).reshape(args.batch_size, 240, 256, 3)
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
        output = Conv2D(16, kernel_size=3, activation="relu")(inputs)
        output = MaxPooling2D(pool_size=(2, 2))(output)
        output = Conv2D(32, kernel_size=3, activation="relu")(output)
        output = MaxPooling2D(pool_size=(2, 2))(output)
        output = Conv2D(64, kernel_size=3, activation="relu")(output)
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
        state = state/255
        if self.epsilon > args.eps_min:
            self.epsilon *= args.eps_decay
        else:
            self.epsilon = args.eps_min
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)
        else:
            q_value = self.predict(state)[0]
            return np.argmax(q_value)
    def train(self, states, targets):
        self.model.fit(states, targets, epochs=1, verbose=1)
    def summary(self):
        return self.model.summary()
    def save(self, path):
        self.model.save(path)
    
class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.n
        self.model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_update()
        self.buffer = ReplayBuffer()
    def target_update(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)
    def replay(self):
        for _ in range(10):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.target_model.predict(states)
            next_q_values = self.target_model.predict(next_states).max(axis=1)
            targets[range(args.batch_size), actions] = rewards + (1-done) * next_q_values * args.gamma
            self.model.train(states, targets)
    def train(self, max_episodes=1000):
        # dict_action = {0:"NOOP", 1:"A", 2:"B", 3:"right", 4:"left", 5:"down"}
        for episode in range(max_episodes):
            done, total_reward = False, 0
            state = self.env.reset()
            while not done:
                action = self.model.get_action(state)
                # print(dict_action[action])
                next_state, reward, done, _ = self.env.step(action)
                self.env.render()
                self.buffer.put(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
            if self.buffer.size() >= args.batch_size:
                self.replay()
            self.target_update()
            self.save()
            print('EP{} : Reward={}'.format(episode, total_reward))
        self.env.close()
    def save(self, path="tetris.ckpt"):
        self.model.save(path)
    def load(self, path="tetris.ckpt"):
        self.model = load_model(path)
        self.target_model = load_model(path)
         
def main():
    env = gym_tetris.make('TetrisA-v3')
    env = JoypadSpace(env, MOVEMENT)
    agent = Agent(env)
    # agent.load()
    agent.train(max_episodes=1000)
    
if __name__ == "__main__":
    main()