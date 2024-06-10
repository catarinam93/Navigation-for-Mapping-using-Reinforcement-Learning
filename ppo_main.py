'''This script focuses on training a Proximal Policy Optimization (PPO) agent using Stable Baselines3 library.
It begins by setting up the simulation environment with Supervisor object and enabling various devices such
as Lidar, Compass, GPS, and Touch Sensor. The environment uses a deterministic occupancy grid map to represent
the robot's surroundings. After obtaining initial sensor readings and creating a transformation matrix for the
robot's pose, LiDAR data is processed to update the map. The script then proceeds to train the PPO agent using
the custom environment and saves the trained model at specified intervals.'''

from controller import Lidar, Compass, GPS, TouchSensor, Supervisor
from stable_baselines3 import PPO
from environment import *
from map import *
import gymnasium as gym
import os
import torch

print(torch.cuda.is_available())


def main():
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

    # Read the LiDAR data and update the map
    point_cloud = lidar.getPointCloud()
    valid_points = [(point.x, point.y) for point in point_cloud if
                    not (math.isnan(point.x) or math.isnan(point.y) or math.isnan(point.z))]
    map.update_map(supervisor_tf, valid_points)

    # ----------------------------------- PPO ---------------------------------------------

    # Define directories for saving models and logging
    models_dir = "models/PPO"
    logdir = "tensorboard"

    # Create directories if they don't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Register and create the custom environment
    gym.register(id='CustomEnv-ppo', entry_point=lambda: Environment(supervisor))
    env = gym.make('CustomEnv-ppo')

    # Define training parameters
    TIMESTEPS = 10000
    iters = 0
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

    for i in range(iters, 100):
        # Check if there are previously trained models
        model_path = f"{models_dir}/{TIMESTEPS * i}.zip"
        if os.path.exists(model_path):
            print(f"Loading the trained model from {model_path}")
            model = PPO.load(model_path, env)
        else:
            # Train the model
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="ppo")
            # Save the model after each iteration
            model.save(f"{models_dir}/{TIMESTEPS * (i + 1)}")


def contains_nan(values):
    # Check if the list contains NaN values
    return any(math.isnan(value) for value in values)


if __name__ == "__main__":
    main()
