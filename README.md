# rl-algorithms

This repository contains the implementation of various reinforcement learning algorithms in PyTorch. These algorithms can be used to train agents to solve a wide range of tasks, from playing games to real-world applications such as robotics.

This repository depends on the following frameworks:

- **[WeightAndBiases](https://www.wandb.com/)**: a tool for tracking and visualizing machine learning experiments.
- **[Hydra](https://hydra.cc/)**: a framework for managing complex applications, with support for configuration, logging, and more.
- **[dm_env](https://github.com/deepmind/dm_env)**: a package for defining and using reinforcement learning environments in DeepMind's ecosystem.

## Algorithms

The following algorithms are included in this repository:

- **Deep Q-learning**: a variant of Q-learning that uses a deep neural network to represent the action-value function, allowing for better generalization and the ability to handle high-dimensional state spaces. Q-learning is a model-free algorithm for learning a policy that maps states to actions, using the Bellman equation to update the action-value function.


