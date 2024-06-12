# README

## Description
Assignment of the course Introduction to Intelligent Robotics (3rd year, 2nd semester)

## Files


- **environment.py**: Definition of the simulation environment where the navigation agent interacts.
- **map.py**: Includes class DeterministicOccupancyGrid that represents a deterministic occupancy grid map.
- **ppo_main.py**: Implementation of the PPO (Proximal Policy Optimization) algorithm for training the navigation agent.
- **a2c_main.py**: Implementation of the A2C (Advantage Actor-Critic) algorithm for training the navigation agent.
- **sac_main.py**: Implementation of the SAC (Soft Actor-Critic) algorithm for training the navigation agent.
- **settings.py**: Set of reward constants for RL's algorithms.
- **occupancy_grid.py**: Occupancy Grid base class (By: Gonçalo Leão).
- **transformations.py**: Functions for working with 3D transformations (By: Gonçalo Leão).
- **utils.py**: Functions for Webots and for working with the epuck robot (By: Gonçalo Leão).
- **test.py**: Script for testing the performance of trained models in a given environment.
- **worlds/**: Folder containing the environment worlds for Webots simulation.
- **instructions.txt**: File containing instructions to run the code.
- **images_timesteps.py**: Script to calculate the average number of timesteps for each method and map.