# code from rllib document

import ray
from ray import train, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from tqdm import tqdm

ray.init()

# configure the algorithm
config = (
  PPOConfig()
  .environment("CartPole-v1")
  .rollouts(num_rollout_workers=2)
  .framework("torch")
  .resources(num_gpus=1)
  .training(
    lr=tune.grid_search([0.01, 0.003, 0.001, 0.0003, 0.0001]), 
    model={"fcnet_hiddens": [64, 64]}
  )
  .evaluation(evaluation_num_workers=1)
)

# set tuner
tuner = tune.Tuner(
  "PPO",
  run_config=train.RunConfig(
    stop={"episode_reward_mean": 200},
    checkpoint_config=train.CheckpointConfig(checkpoint_at_end=True)
  ),
  param_space=config
)

# tune
result = tuner.fit()
best_result = result.get_best_result(
  metric="episode_reward_mean",
  mode="max"
)
best_checkpoint = best_result.checkpoint
