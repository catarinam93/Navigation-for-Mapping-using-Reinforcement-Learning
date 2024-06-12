# README

## Description
Assignment of the course Introduction to Intelligent Robotics (3rd year, 2nd semester)

## Files


- **a2c_main.py**: Implementation of the A2C (Advantage Actor-Critic) algorithm for training the navigation agent.
- **ppo_main.py**: Implementation of the PPO (Proximal Policy Optimization) algorithm for training the navigation agent.
- **sac_main.py**: Implementation of the SAC (Soft Actor-Critic) algorithm for training the navigation agent.
- **td3_main.py**: Implementation of the TD3 (Twin Delayed Deep Deterministic Policy Gradient) algorithm for training the navigation agent.
- **environment.py**: Definition of the simulation environment where the navigation agent interacts.
- **map.py**: Includes class DeterministicOccupancyGrid that represents a deterministic occupancy grid map.
- **occupancy_grid.py**: Occupancy Grid base class (By: Gonçalo Leão).
- **transformations.py**: Functions for working with 3D transformations (By: Gonçalo Leão).
- **utils.py**: Functions for Webots and for working with the epuck robot (By: Gonçalo Leão).
- **images_timesteps.py**: Script to calculate the average number of timesteps for each method and map.
- **instructions.txt**: File containing instructions to run the code.
- **settings.py**: Set of reward constants for RL's algorithms.
- **test.py**: Script for testing the performance of trained models in a given environment.
- **maps_images/**: Folder containing images of the generated maps while training.
- **models/**: Folder containing trained models.
- **tensorboard/**: Folder containing TensorBoard logs.
- **test_maps/**: Folder containing generated maps during test.