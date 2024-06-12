# README

## Navigation-for-Mapping-using-Reinforcement-Learning
Assignment of the course Introduction to Intelligent Robotics (3rd year, 2nd semester)

## A little context
### Overview
The aim of this project was to develop a mobile robot that must build a map of an unknown environment using a LiDAR reading. The goal is to use reinforcement learning (RL) models to control how the robot should move so that all of the environment gets mapped. It is assumed that the environment is static (i.e. no moving objects). 

### The environment chosen's characteristics
#### Action Space
Actions are the linear and angular velocitites.
#### Rewards
Reward is given for every map completly explored and for neutral behaviour. 
Penalties are given if there is a collision or if the robot gets to close to the walls
#### States
The initial state places the robot on a random wall-free spot on the world (to avoid overfitting). Episodes end if the robot can map the entire world.
#### Percepts (Observations)
Observations include various details such as collision sensor's and Lidar sensor's output and the occupancy map.
#### RL algorithms chosen
* PPO
* SAC
* A2C
* TD3

### Simulation Maps
![Simulation Maps](images/simulation_maps/map0.jpg)
![Simulation Maps](images/simulation_maps/map1.jpg)
![Simulation Maps](images/simulation_maps/map2.jpg)
![Simulation Maps](images/simulation_maps/map3.jpg)
![Simulation Maps](images/simulation_maps/map4.jpg)
![Simulation Maps](images/simulation_maps/map5.jpg)

### Training Process

### Experimental Results - on tensorboard
![Experimental Results](images/experimental_results.png)

#### Test Results
After training, tests were wade on a map (map6) of increased difficulty. The tests were conducted only with PPO and A2C once these were the only algorithms which manage to complete training. The following maps are the results of that test phase.
##### PPO
![Test](images/test_maps/PPO/9.png)
##### A2C
![Test](images/test_maps/A2C/9.png)

### Conclusion


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
