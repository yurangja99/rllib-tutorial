# code from rllib document

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from tqdm import tqdm

# configure the algorithm
config = (
  PPOConfig()
  .environment("CartPole-v1")
  .rollouts(num_rollout_workers=2)
  .framework("torch")
  .resources(num_gpus=1)
  .training(model={"fcnet_hiddens": [64, 64]})
  .evaluation(evaluation_num_workers=1)
)

# build the algorithm with the config
algorithm = config.build()

# train
for _ in tqdm(range(10)):
  last_train_status = algorithm.train()

print(pretty_print(last_train_status))

# evaluate
print(pretty_print(algorithm.evaluate()))
