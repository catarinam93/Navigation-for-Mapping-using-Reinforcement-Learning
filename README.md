# Navigation-for-Mapping-using-Reinforcement-Learning
# README

## Description
Assignment of the course Introduction to Intelligent Robotics (3rd year, 2nd semester)

## Files

- **environment.py**: Definition of the simulation environment where the navigation agent interacts.
- **map.py**: Includes class DeterministicOccupancyGrid that represents a deterministic occupancy grid map.
- **ppo_main.py**: Implementation of the PPO (Proximal Policy Optimization) algorithm for training the navigation agent.
- **sac_main.py**: Implementation of the SAC (Soft Actor-Critic) algorithm for training the navigation agent.
- **a2c_main.py**: Implementation of the A2C (Advantage Actor-Critic) algorithm for training the navigation agent.
- **settings.py**: Set of reward constants for RL's algorithms.
- **occupancy_grid.py**: Occupancy Grid base class (By: Gonçalo Leão).
- **transformations.py**: Functions for working with 3D transformations (By: Gonçalo Leão).
- **utils.py**: Functions for Webots and for working with the epuck robot (By: Gonçalo Leão).
- **worlds/**: Folder containing the environment worlds for Webots simulation.

## Usage Instructions
1. Make sure you have all dependencies installed (listed in the `requirements.txt` file).
2. Run the corresponding script file for the learning algorithm you want to train (`a2c_main.py`, `ppo_main.py`, or `sac_main.py`).
3. Adjust the rewards parameters as necessary in the `settings.py` file.
4. Monitor the training progress and results through the logs generated during execution.

