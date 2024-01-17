# In this code,
# 1. define custom environment
# 2. train PPO for the environment
# 3. run trained agent with the environment

import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
import numpy as np
from tqdm import tqdm
import random

# custom environment class
class SimpleCorridor(gym.Env):
  """
  Variation of the example environment "SimpleCorridor" by rllib document. 
  Objective of the environment is to reach the correct end of the corridor. 
  The correct end is one of two ends. 

  - Observation space: current and previous position
  - Action space: 0 or 1
  - Reward: -0.1 each timestep, 20 when reaches the end. 
  
  Expected optimized rward is 19.1: average of 19.6(lucky) and 18.6(unlucky)

  Source: https://github.com/ray-project/ray/blob/master/rllib/examples/custom_env.py
  """
  def __init__(self, env_config: dict):
    self.end_pos = 10
    self.prev_pos = 5
    self.cur_pos = 5
    self.observation_space = Box(0.0, self.end_pos, shape=(2,), dtype=np.float32)
    self.action_space = Discrete(2)

  def reset(self, seed=0, options=None):
    self.prev_pos = 5
    self.cur_pos = 5
    random.seed(seed)
    self.correct_end = 0 if random.random() < 0.5 else self.end_pos
    return np.array([self.prev_pos, self.cur_pos]), {}

  def step(self, action):
    assert action in [0, 1]

    self.prev_pos = self.cur_pos
    if action == 0 and self.cur_pos > 0:
      self.cur_pos -= 1
    elif action == 1 and self.cur_pos < self.end_pos:
      self.cur_pos += 1
    terminated = truncated = self.cur_pos == self.correct_end

    return np.array([self.prev_pos, self.cur_pos]), 20 if terminated else -0.1, terminated, truncated, {}
  
  def render(self):
    corridor = ["*" for _ in range(11)]
    corridor[self.correct_end] = "G"
    corridor[self.cur_pos] = "A"
    print(" ".join(corridor))

# register custom environment
register_env("simple_corridor", lambda env_config: SimpleCorridor(env_config))

# configure the algorithm
config = (
  PPOConfig()
  .environment("simple_corridor", env_config={})
  .rollouts(num_rollout_workers=2)
  .framework("torch")
  .resources(num_gpus=1)
  .training(model={"fcnet_hiddens": [4, 4]})
  .evaluation(evaluation_num_workers=1)
)

# build the algorithm with the config
algorithm = config.build()

# train
for _ in tqdm(range(50)):
  last_train_status = algorithm.train()

print(pretty_print(last_train_status))

# evaluate
print(pretty_print(algorithm.evaluate()))

# render one episode with the trained algorithm
env = SimpleCorridor(env_config={})
obs, _ = env.reset()
episode_reward = 0.0
done = False
env.render()
while not done:
  action = algorithm.compute_single_action(obs)
  obs, reward, terminated, truncated, _ = env.step(action)
  done = terminated or truncated
  episode_reward += reward
  env.render()
print(f"episode reward: {episode_reward}")
