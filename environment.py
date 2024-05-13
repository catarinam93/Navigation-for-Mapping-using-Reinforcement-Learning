import gymnasium as gym
from controllers.occupancy_grid import OccupancyGrid
from map import DeterministicOccupancyGrid
from controller import Supervisor, Field, Robot
from controllers.utils import cmd_vel
import numpy as np
from typing import List
from settings import *

class Environment(gym.Env):
    def __init__(self, supervisor):
        self.map = None
        self.timesteps = 0
        self.supervisor = supervisor

        self.sensor_bounds = [0, 255]  # Sensor bounds of the environment
        '''self.angle_bounds = [-np.pi, np.pi]  # Angle bounds of the environment
        self.distance_bounds = [0, np.inf]  # Distance bounds of the environment'''

        self.angular_vel_bounds = [-1.0, 1.0]  # Angular Velocity bounds of the environment
        self.linear_vel_bounds = [-1.0, 1.0]  # Linear Velocity bounds of the environment

        self.action_space = gym.spaces.Box(low=np.array([self.linear_vel_bounds[0], self.angular_vel_bounds[0]]),
                                           high=np.array([self.linear_vel_bounds[1], self.angular_vel_bounds[1]]),
                                           dtype=np.float32)  # Action space of the environment

        self.observation_space = gym.spaces.Box(low=np.array([self.sensor_bounds[0]] * 100),
                                                high=np.array([self.sensor_bounds[1]] * 100),
                                                dtype=np.float32)

    def warp_robot(self, supervisor: Supervisor, robot_def_name: str, new_position: (float, float)) -> None:
        robot_node = supervisor.getFromDef(robot_def_name)
        trans_field: Field = robot_node.getField("translation")
        translation: List[float] = [new_position[0], new_position[1], 0]
        trans_field.setSFVec3f(translation)
        robot_node.resetPhysics()


    # Reset the environment
    def reset(self):
        super().reset()  # Reset the environment
        map_origin = (0.0, 0.0)
        map_dimensions = (200, 200)
        map_resolution = 0.01
        self.map = DeterministicOccupancyGrid(map_origin, map_dimensions, map_resolution)  # Resets the map
        initial_position = (0.05, 0.04)
        self.warp_robot(self.supervisor, "EPUCK", initial_position) # fazer reset à orientação do utils
        # Reset all the variables
        self.timesteps = 0  # Reset the timesteps
        self.terminated = False
        self.reward = 0
        observation = self._get_obs()
        # info = self._get_info()
        return observation, None  # None is the info, is mandatory in gym environments

    def calculate_reward(self, num_explored_cells):
        if DeterministicOccupancyGrid.all_cells_explored():
            self.reward += FINAL_REWARD
            self.terminated = True

        elif num_explored_cells == 0:
            self.reward += NULL_REWARD

        else:
            self.reward = NEUTRAL_REWARD

    def _get_obs(self):
        amount_explored = self.map.percentage_explored()
        return amount_explored

    # Step the environment
    def step(self, action):
        linear_velocity = action[0]
        angular_velocity = action[1]

        cmd_vel(self.supervisor, linear_velocity, angular_velocity)

        num_explored_cells = DeterministicOccupancyGrid.update_map()

        self.calculate_reward(num_explored_cells)

        observation = self._get_obs()

        return observation, self.reward, self.terminated  # , info