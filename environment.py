import gymnasium as gym
from map import DeterministicOccupancyGrid
from controller import Supervisor, Field
from controllers.utils import cmd_vel
import numpy as np
from typing import List
from settings import *
from controllers.transformations import create_tf_matrix
import math

class Environment(gym.Env):
    def __init__(self, supervisor):
        self.map = None
        self.timesteps = 0
        self.supervisor = supervisor

        self.sensor_bounds = [0, 255]  # Sensor bounds of the environment
        self.angular_vel_bounds = [-1.0, 1.0]  # Angular Velocity bounds of the environment
        self.linear_vel_bounds = [-1.0, 1.0]  # Linear Velocity bounds of the environment

        self.action_space = gym.spaces.Box(low=np.array([self.linear_vel_bounds[0], self.angular_vel_bounds[0]]),
                                           high=np.array([self.linear_vel_bounds[1], self.angular_vel_bounds[1]]),
                                           dtype=np.float32)  # Action space of the environment

        self.observation_space = gym.spaces.Box(low=np.array([self.sensor_bounds[0]] * 100),
                                                high=np.array([self.sensor_bounds[1]] * 100),
                                                dtype=np.float32)

    # Reset the environment
    def reset(self, **kwargs):
        super().reset(**kwargs)  # Reset the environment with the provided kwargs
        map_origin = (0.0, 0.0)
        map_dimensions = (200, 200)
        map_resolution = 0.01
        self.map = DeterministicOccupancyGrid(map_origin, map_dimensions, map_resolution)  # Resets the map
        initial_position = (0.05, 0.04)
        self.warp_robot(self.supervisor, "EPUCK", initial_position)  # fazer reset à orientação do utils
        # Reset all the variables
        self.timesteps = 0  # Reset the timesteps
        self.terminated = False
        self.reward = 0
        observation = self._get_obs()
        # info = self._get_info()
        return observation, {}  # None is the info, is mandatory in gym environments

    # Step the environment
    def step(self, action):
        linear_velocity = action[0]
        angular_velocity = action[1]

        cmd_vel(self.supervisor, linear_velocity, angular_velocity)

        # Obtain supervisor's position from GPS
        gps_readings = self.supervisor.getDevice('gps').getValues()
        supervisor_position = (gps_readings[0], gps_readings[1])

        # Obtain supervisor's orientation from Compass
        compass_readings = self.supervisor.getDevice('compass').getValues()
        supervisor_orientation = math.atan2(compass_readings[0], compass_readings[1])

        supervisor_tf: np.ndarray = create_tf_matrix((supervisor_position[0], supervisor_position[1], 0.0),
                                                     supervisor_orientation)

        # Obtain lidar points
        lidar = self.supervisor.getDevice('lidar')
        point_cloud = lidar.getPointCloud()
        valid_points = [point for point in point_cloud if
                        not (math.isnan(point.x) or math.isnan(point.y) or math.isnan(point.z))]

        num_explored_cells = self.map.update_map(supervisor_tf, valid_points)

        self.calculate_reward(num_explored_cells)

        observation = self._get_obs()

        return observation, self.reward, self.terminated, False, {}

    def warp_robot(self, supervisor: Supervisor, robot_def_name: str, new_position: (float, float)) -> None:
        robot_node = supervisor.getFromDef(robot_def_name)
        trans_field: Field = robot_node.getField("translation")
        translation: List[float] = [new_position[0], new_position[1], 0]
        trans_field.setSFVec3f(translation)
        robot_node.resetPhysics()


    def calculate_reward(self, num_explored_cells):
        if self.map.all_cells_explored():
            self.reward += FINAL_REWARD
            self.terminated = True

        elif num_explored_cells == 0:
            self.reward += NULL_REWARD

        else:
            self.reward = NEUTRAL_REWARD

    def _get_obs(self):
        amount_explored = self.map.percentage_explored()
        return amount_explored

