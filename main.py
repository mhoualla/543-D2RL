import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

class GridEnv(gym.Env):
    def __init__(self, size=5):
        super(GridEnv, self).__init__()
        self.size = size
        self.state = [0, 0] # starting position is the top left corner
        self.goal = [size - 1, size - 1]  # goal is the bottom right corner
        self.danger_zone = [[2, 2]]
        self.observation_space = gym.spaces.Box(low=0, high=size-1, shape=(2,), dtype=np.int32)
        self.action_space = gym.spaces.Discrete(4) 

    def reset(self):
        self.state = [0, 0]
        return np.array(self.state)

    def step(self, action):
        if action == 0 and self.state[0] > 0:
            self.state[0] -= 1
        elif action == 1 and self.state[0] < self.size - 1:
            self.state[0] += 1
        elif action == 2 and self.state[1] > 0:
            self.state[1] -= 1
        elif action == 3 and self.state[1] < self.size - 1:
            self.state[1] += 1

        reward = -1 
        done = False
        if self.state == self.goal:
            reward = 10 
            done = True
        elif self.state in self.danger_zone:
            reward = -10  
            done = True  

        return np.array(self.state), reward, done, {}

env = DummyVecEnv([lambda: GridEnv(size=5)])

model_drl = PPO("MlpPolicy", env, verbose=1)
model_drl.learn(total_timesteps=5000)

class DenseGridEnv(GridEnv):
    def step(self, action):
        state, reward, done, info = super().step(action)

        if state.tolist() not in self.danger_zone and state.tolist() != self.goal:
            reward = 0 
        
        return state, reward, done, info

d2rl_env = DummyVecEnv([lambda: DenseGridEnv(size=5)])
model_d2rl = PPO("MlpPolicy", d2rl_env, verbose=1)
model_d2rl.learn(total_timesteps=5000)

def evaluate_model(env, model, episodes=10):
    rewards = []
    for _ in range(episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards)

drl_score = evaluate_model(env, model_drl)
d2rl_score = evaluate_model(d2rl_env, model_d2rl)

print(f"Average Reward with DRL: {drl_score}")
print(f"Average Reward with D2RL: {d2rl_score}")

def visualize_path(env, model):
    state = env.reset()
    path = [state[0]] 
    done = False
    while not done:
        action, _ = model.predict(state)
        state, _, done, _ = env.step(action)
        path.append(state[0])  


    grid_size = env.envs[0].size 
    grid = np.zeros((grid_size, grid_size))
    for pos in path:
        grid[pos[0], pos[1]] = 1
    plt.imshow(grid, cmap="Blues", origin="upper")
    plt.show()

print("DRL Path:")
visualize_path(env, model_drl)

print("D2RL Path:")
visualize_path(d2rl_env, model_d2rl)
