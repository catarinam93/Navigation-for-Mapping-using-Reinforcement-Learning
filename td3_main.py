'''This script demonstrates the usage of the Stable Baselines3 library for training a Soft Actor-Critic (SAC) agent
in a custom robotics environment. It starts by setting up the simulation environment with a Supervisor object and
initializing devices such as Lidar, Compass, GPS, and Touch Sensor. The environment uses a deterministic occupancy
grid map to represent the robot's surroundings. After obtaining initial sensor readings and creating a transformation
matrix for the robot's pose, LiDAR data is processed to update the map. The script then proceeds to train the SAC agent
using the custom environment and saves the trained model at specified intervals.'''

from stable_baselines3 import TD3
from controller import Lidar, Compass, GPS, TouchSensor
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


    # ----------------------------------- TD3 ---------------------------------------------

    # Define directories for saving models and logging
    models_dir = "models/TD3"
    logdir = "tensorboard"
    mapsdir = "maps_images/TD3/map0/iter"
    final_mapsdir = "maps_images/TD3/map0/final_maps" # change map_x depending on the used world

    # Create directories if they don't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if not os.path.exists(mapsdir):
        os.makedirs(mapsdir)

    if not os.path.exists(final_mapsdir):
        os.makedirs(final_mapsdir)

    # Register and create the custom environment
    gym.register(id='CustomEnv-td3', entry_point=lambda: Environment(supervisor, final_mapsdir))
    env = gym.make('CustomEnv-td3')

    # Define training parameters
    TIMESTEPS = 50000
    iters = 0

    model = TD3("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

    for i in range(iters, 5):
        # Check if there are previously trained models
        model_path = f"{models_dir}/{i}.zip"
        if os.path.exists(model_path):
            print(f"Loading the trained model from {model_path}")
            model = TD3.load(model_path, env)
        # Train the model
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="td3")
        # Save the model after each iteration
        model.save(f"{models_dir}/{(i + 1)}")
        # Visualize and save the map
        map.plot_grid(save_path=f"{mapsdir}/{(i + 1)}.png")


def contains_nan(values):
    # Check if the list contains NaN values
    return any(math.isnan(value) for value in values)


if __name__ == "__main__":
    main()