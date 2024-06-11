'''This script tests the performance of PPO, A2C, and TD3 trained models in a given environment.
It loads the models, runs them in the environment, and generates maps of the robot's surroundings.'''

from controller import Lidar, Compass, GPS, TouchSensor, Supervisor
from stable_baselines3 import PPO, A2C, TD3
from environment import *
from map import *
import gymnasium as gym
import os
import torch
import math
import numpy as np

print(torch.cuda.is_available())

def test_model(model_class, model_path, env_id, map_save_path):
    # Create a Supervisor object
    supervisor: Supervisor = Supervisor()
    # Get the simulation timestep
    timestep: int = int(supervisor.getBasicTimeStep())

    # Set map parameters
    map_origin = (0.0, 0.0)
    map_dimensions = (1, 1)
    map_resolution = 0.01

    # Initialize the deterministic occupancy grid map
    map: DeterministicOccupancyGrid = DeterministicOccupancyGrid(map_origin, map_dimensions, map_resolution)

    # Initialize and enable the Lidar device
    lidar: Lidar = supervisor.getDevice('lidar')
    lidar.enable(timestep)
    lidar.enablePointCloud()

    # Initialize and enable the Touch Sensor device
    touch_sensor: TouchSensor = supervisor.getDevice('touch sensor')
    touch_sensor.enable(timestep)

    # Initialize and enable the Compass device
    compass: Compass = supervisor.getDevice('compass')
    compass.enable(timestep)

    # Initialize and enable the GPS device
    gps: GPS = supervisor.getDevice('gps')
    gps.enable(timestep)

    # Read initial GPS and compass values
    gps_readings: [float] = gps.getValues()
    compass_readings: [float] = compass.getValues()

    # Wait until valid GPS and compass readings are obtained
    while contains_nan(gps_readings) or contains_nan(compass_readings):
        gps_readings: [float] = gps.getValues()
        compass_readings: [float] = compass.getValues()
        supervisor.step(timestep)

    # Print the GPS and compass readings
    print("gps:", gps_readings)
    print("compass:", compass_readings)

    # Extract the robot's position and orientation
    supervisor_position: (float, float) = (gps_readings[0], gps_readings[1])
    supervisor_orientation: float = math.atan2(compass_readings[0], compass_readings[1])

    # Print the robot's position and orientation
    print("robot_position:", supervisor_position)
    print("robot_orientation:", supervisor_orientation)

    # Create the transformation matrix for the robot's pose
    supervisor_tf: np.ndarray = create_tf_matrix((supervisor_position[0], supervisor_position[1], 0.0),
                                                 supervisor_orientation)

    # Load the trained model
    model = model_class.load(model_path)

    # Register and create the custom environment
    gym.register(id=env_id, entry_point=lambda: Environment(supervisor, map_save_path))
    env = gym.make(env_id)

    # Run the environment with the loaded model
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)

        # Read the LiDAR data and update the map
        point_cloud = lidar.getPointCloud()
        valid_points = [(point.x, point.y) for point in point_cloud if
                        not (math.isnan(point.x) or math.isnan(point.y) or math.isnan(point.z))]
        map.update_map(supervisor_tf, valid_points)

        supervisor.step(timestep)

    # Visualize and save the map
    map.plot_grid(save_path=map_save_path)

def contains_nan(values):
    # Check if the list contains NaN values
    return any(math.isnan(value) for value in values)

if __name__ == "__main__":
    # Define model directories
    models_dir = "models"
    maps_dir = "test_maps"

    if not os.path.exists(maps_dir):
        os.makedirs(maps_dir)

    # Test PPO model
    test_model(PPO, f"{models_dir}/PPO/10.zip", 'CustomEnv-ppo', f"{maps_dir}/ppo_map.png")

    # Test A2C model
    test_model(A2C, f"{models_dir}/A2C/10.zip", 'CustomEnv-a2c', f"{maps_dir}/a2c_map.png")

    # Test TD3 model
    test_model(TD3, f"{models_dir}/TD3/10.zip", 'CustomEnv-td3', f"{maps_dir}/td3_map.png")
