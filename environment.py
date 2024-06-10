'''This script defines a custom environment Environment for reinforcement learning tasks in robotics simulation.
It leverages Webots for simulation and control. The environment includes functionalities for initializing the map,
updating it based on sensor readings, and calculating rewards for agent actions. Additionally, it provides methods
for warping the robot to safe positions, finding safe regions, and selecting random positions within them.
The observation space encompasses LiDAR readings, map state, and touch sensor readings, while the action space
defines the bounds for linear and angular velocities.'''

import numpy as np
from typing import List, Tuple
from map import DeterministicOccupancyGrid
from controller import Supervisor, Field
from utils import cmd_vel
from settings import *
from transformations import create_tf_matrix
import math
import random
import gymnasium as gym

class Environment(gym.Env):
    def __init__(self, supervisor):
        # Initialize the environment with a supervisor object
        self.map = None  # Initialize the map to None
        self.timesteps = 0  # Initialize the timestep counter
        self.supervisor = supervisor  # Store the supervisor object

        self.terminated = False  # Set the termination flag to False
        self.reward = 0  # Initialize the reward

        # Define the bounds for angular and linear velocities
        self.angular_vel_bounds = [0, 0.1]  # Angular velocity bounds of the environment
        self.linear_vel_bounds = [0, 0.1]  # Linear velocity bounds of the environment

        # Define the action space of the environment
        self.action_space = gym.spaces.Box(low=np.array([self.linear_vel_bounds[0], self.angular_vel_bounds[0]]),
                                           high=np.array([self.linear_vel_bounds[1], self.angular_vel_bounds[1]]),
                                           dtype=np.float32)

        num_lidar_readings = 200  # Number of LiDAR readings
        map_state_size = 10000  # Size of the map state

        # Minimum values of observations
        obs_low = np.concatenate([
            np.zeros(num_lidar_readings),  # LiDAR readings (min: 0)
            np.zeros(map_state_size),  # Minimum occupancy map (0 represents unexplored)
            [0.0]  # Touch sensor reading (0.0 represents no collision)
        ])

        # Maximum values of observations
        obs_high = np.concatenate([
            np.full(num_lidar_readings, 100),  # Maximum LiDAR readings (adjust as necessary)
            np.full(map_state_size, 1.0),  # Maximum occupancy map (1 represents explored and occupied)
            [1.0]  # Touch sensor reading (1.0 represents collision)
        ])

        # Define the observation space
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Add the collection of wall positions
        self.wall_positions = self.get_wall_positions()

    def get_wall_positions(self):
        # Function to get the positions of walls in the environment
        wall_positions = []
        root = self.supervisor.getRoot()  # Get the root of the supervisor
        children_field = root.getField("children")  # Get the children field of the root
        for i in range(children_field.getCount()):  # Iterate over all children
            node = children_field.getMFNode(i)  # Get the node at index i
            if node.getTypeName() == "Solid":  # Check if the node is a solid object
                position = node.getField("translation").getSFVec3f()  # Get the position of the wall
                wall_positions.append((position[0], position[1]))  # Add the position to the list
        return wall_positions

    def find_safe_region(self):
        # Function to find safe regions where the robot can be placed
        safe_region = []
        x = 0.0
        while x <= 1.0:  # Iterate over the x dimension
            y = 0
            while y <= 1:  # Iterate over the y dimension
                if not self.map.is_wall((x, y)) and (x, y) not in self.wall_positions:
                    # Check if the position is not a wall and not in the list of wall positions
                    safe_region.append((x, y))  # Add the position to the safe region list
                y += 0.01  # Increment the y coordinate
            x += 0.01  # Increment the x coordinate
        return safe_region

    def select_random_position(self, safe_region):
        # Function to select a random position within the safe region
        initial_position = random.choice(safe_region)
        return initial_position

    # Reset the environment
    def reset(self, **kwargs):
        super().reset(**kwargs)  # Call the reset method of the parent class
        map_origin = (0.0, 0.0)  # Define the origin of the map
        map_dimensions = (1, 1)  # Define the dimensions of the map
        map_resolution = 0.01  # Define the resolution of the map
        self.map = DeterministicOccupancyGrid(map_origin, map_dimensions, map_resolution)  # Create a new map

        # Find a safe region and select a random position within it
        safe_region = self.find_safe_region()
        initial_position = self.select_random_position(safe_region)

        # Make the robot start at the random position
        self.warp_robot(self.supervisor, "EPUCK", initial_position)

        # Reset all the variables
        self.timesteps = 0  # Reset the timestep counter
        self.terminated = False  # Reset the termination flag
        self.reward = 0  # Reset the reward

        observation = self._get_obs()  # Get the initial observation

        return observation, {}  # Return the initial observation and an empty info dictionary

    # Step the environment
    def step(self, action):
        linear_velocity = action[0]  # Extract the linear velocity from the action
        angular_velocity = action[1]  # Extract the angular velocity from the action

        # Make the robot move
        cmd_vel(self.supervisor, linear_velocity, angular_velocity)
        self.supervisor.step(int(self.supervisor.getBasicTimeStep()))  # Step the simulation

        # Update the position and orientation of the supervisor
        gps_readings = self.supervisor.getDevice('gps').getValues()  # Get GPS readings
        supervisor_position = (gps_readings[0], gps_readings[1])  # Extract position from GPS readings
        compass_readings = self.supervisor.getDevice('compass').getValues()  # Get compass readings
        supervisor_orientation = math.atan2(compass_readings[0], compass_readings[1])  # Calculate orientation
        supervisor_tf = create_tf_matrix((supervisor_position[0], supervisor_position[1], 0.0), supervisor_orientation)

        # Get the observation
        observation = self._get_obs()

        # Get the point cloud data from the LiDAR observation (first 200 elements)
        point_cloud = observation[:200].reshape(-1, 2)  # Reshape into pairs of coordinates (x, y)

        # Update the map
        valid_points = [(x, y) for x, y in point_cloud if not (math.isnan(x) or math.isnan(y))]

        num_explored_cells = self.map.update_map(supervisor_tf, valid_points)  # Update the map with valid points

        collision = observation[-1]  # Get the collision status from the observation

        # Calculate the reward
        self.calculate_reward(num_explored_cells, valid_points, supervisor_position, collision)

        return observation, self.reward, self.terminated, False, {}  # Return the observation, reward, terminated flag, truncated flag, and info dictionary

    def warp_robot(self, supervisor: Supervisor, robot_def_name: str, new_position: (float, float)) -> None:
        # Function to warp the robot to a new position
        robot_node = supervisor.getFromDef(robot_def_name)  # Get the robot node by its DEF name
        if robot_node is None:
            raise ValueError(f"Robot with DEF name '{robot_def_name}' not found.")  # Raise an error if the robot is not found
        trans_field: Field = robot_node.getField("translation")  # Get the translation field of the robot
        translation: List[float] = [new_position[0], new_position[1], 0]  # Define the new position
        trans_field.setSFVec3f(translation)  # Set the new position
        robot_node.resetPhysics()  # Reset the physics of the robot to apply the new position

    def calculate_reward(self, num_explored_cells, point_cloud, robot_position, collision):
        # Function to calculate the reward
        if self.map.all_cells_explored():  # Check if all cells are explored
            self.reward += FINAL_REWARD  # Add the final reward
            self.terminated = True  # Set the termination flag to True
        elif num_explored_cells == 0:  # Check if no cells were explored
            self.reward += NULL_REWARD  # Add the null reward
        elif any(np.linalg.norm([point[0] - robot_position[0], point[1] - robot_position[1]]) < 0.1 for point in point_cloud):
            # Check if any point is too close to the robot
            self.reward += PENALTY_REWARD  # Add the penalty reward
        elif collision == 1.0:  # Check if there was a collision
            self.reward += COLLISION_REWARD  # Add the collision reward
        else:
            self.reward = NEUTRAL_REWARD  # Add the neutral reward

    def _get_obs(self):
        # Function to get the observation
        lidar = self.supervisor.getDevice('lidar')  # Get the LiDAR device
        lidar_data = lidar.getPointCloud()  # Get the point cloud data from the LiDAR

        processed_point_cloud = []
        for point in lidar_data:  # Process the LiDAR data
            processed_point_cloud.append(point.x)  # Append the x coordinate
            processed_point_cloud.append(point.y)  # Append the y coordinate

        # Get the state of the map
        map_state = self.map.get_flattened_state()
        map_state_float = np.array(map_state, dtype=np.float32)  # Convert the map state to a float array

        # Define the replacement values for inf and -inf
        max_value = 1.0  # Maximum expected value (1.0 is the length of the world)
        min_value = -1.0  # Minimum expected value

        processed_point_cloud_float = np.array(processed_point_cloud, dtype=np.float32)  # Convert the point cloud to a float array

        # Replace inf and -inf with the defined maximum and minimum values
        processed_point_cloud_array = np.where(processed_point_cloud_float == float('inf'), max_value, processed_point_cloud_float)
        processed_point_cloud_array = np.where(processed_point_cloud_array == float('-inf'), min_value, processed_point_cloud_array)

        # Get the touch sensor reading and convert to a 1D array
        touch_sensor = self.supervisor.getDevice('touch sensor')
        touch_sensor_reading = np.array([touch_sensor.getValue()], dtype=np.float32)

        # Concatenate all readings into a single observation array
        observation = np.concatenate([processed_point_cloud_array, map_state_float, touch_sensor_reading]).astype(np.float32)

        return observation
