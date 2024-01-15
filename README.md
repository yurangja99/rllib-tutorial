# rllib-tutorial

## RLlib documents
- https://docs.ray.io/en/latest/rllib/index.html
- https://github.com/ray-project/ray/tree/master/rllib

## Features of RLlib
- support PyTorch and Tensorflow
- highly distributed learning
- multi-agent RL
- external simulators
- offline RL and imitation learning/behavior cloning

## Visualization
In Windows, `[PATH]` is `C:\Users\<USER>\ray_results\<DIR>`
```commandline
tensorboard --logdir [PATH]
```

## Todo
- [ ] Custom Environment ([reference](https://docs.ray.io/en/latest/rllib/rllib-env.html#configuring-environments))
- [ ] Custom Policy ([reference](https://docs.ray.io/en/latest/rllib/rllib-concepts.html))
- [ ] Imitation Learning ([reference1](https://github.com/ray-project/ray/blob/master/rllib/examples/custom_model_loss_and_metrics.py), [reference2](https://docs.ray.io/en/latest/rllib/rllib-offline.html#input-pipeline-for-supervised-losses))
