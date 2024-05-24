import gymnasium as gym
from map import DeterministicOccupancyGrid
from controller import Supervisor, Field
from utils import cmd_vel
import numpy as np
from typing import List
from settings import *
from transformations import create_tf_matrix
from controller import LidarPoint
import math

class Environment(gym.Env):
    def __init__(self, supervisor):
        self.map = None
        self.timesteps = 0
        self.supervisor = supervisor

        self.amount_exp = [0, 100]
        self.sensor_bounds = [0, 100]  # Sensor bounds of the environment
        self.angular_vel_bounds = [0, 0.1]  # Angular Velocity bounds of the environment
        self.linear_vel_bounds = [0, 0.1]  # Linear Velocity bounds of the environment

        self.action_space = gym.spaces.Box(low=np.array([self.linear_vel_bounds[0], self.angular_vel_bounds[0]]),
                                           high=np.array([self.linear_vel_bounds[1], self.angular_vel_bounds[1]]),
                                           dtype=np.float32)  # Action space of the environment

        # Observation space now needs to be a flat array, let's say we have a fixed size for the point cloud
        point_cloud_size = 360  # Assuming 360 points in the Lidar
        self.observation_space = gym.spaces.Box(low=np.array([self.amount_exp[0]] + [self.sensor_bounds[0]] * point_cloud_size),
                                                high=np.array([self.amount_exp[1]] + [self.sensor_bounds[1]] * point_cloud_size),
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

        # Define a velocidade do robô
        cmd_vel(self.supervisor, linear_velocity, angular_velocity)
        self.supervisor.step()

        # Atualiza a posição e orientação do supervisor
        gps_readings = self.supervisor.getDevice('gps').getValues()
        supervisor_position = (gps_readings[0], gps_readings[1])
        compass_readings = self.supervisor.getDevice('compass').getValues()
        supervisor_orientation = math.atan2(compass_readings[0], compass_readings[1])
        supervisor_tf = create_tf_matrix((supervisor_position[0], supervisor_position[1], 0.0), supervisor_orientation)

        # Obtém a observação
        observation = self._get_obs()
        point_cloud = observation[1:]

        # Atualiza o mapa
        valid_points = []
        for i in range(0, len(point_cloud), 3):
            x, y, z = point_cloud[i], point_cloud[i + 1], point_cloud[i + 2]
            if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
                point = LidarPoint(0,0)
                print(dir(point))
                point.x = x
                point.y = y
                point.z = z
                valid_points.append(point)

        # Atualiza o mapa
        num_explored_cells = self.map.update_map(supervisor_tf, valid_points)

        # Calcula a recompensa
        self.calculate_reward(num_explored_cells, point_cloud, supervisor_position)

        return observation, self.reward, self.terminated, False, {}

    def warp_robot(self, supervisor: Supervisor, robot_def_name: str, new_position: (float, float)) -> None:
        robot_node = supervisor.getFromDef(robot_def_name)
        trans_field: Field = robot_node.getField("translation")
        translation: List[float] = [new_position[0], new_position[1], 0]
        trans_field.setSFVec3f(translation)
        robot_node.resetPhysics()

    def calculate_reward(self, num_explored_cells, point_cloud, robot_position):
        if self.map.all_cells_explored():
            self.reward += FINAL_REWARD
            self.terminated = True
        elif num_explored_cells == 0:
            self.reward += NULL_REWARD
        elif any(np.linalg.norm([point.x - robot_position[0], point.y - robot_position[1]]) < 0.1 for point in point_cloud):
            self.reward += PENALTY_REWARD
        else:
            self.reward = NEUTRAL_REWARD

    def _get_obs(self):
        # Obtém a leitura do lidar
        lidar = self.supervisor.getDevice('lidar')
        point_cloud = lidar.getPointCloud()
        amount_explored = self.map.percentage_explored()

        # Flatten the point cloud data and limit its size
        flattened_point_cloud = [value for point in point_cloud for value in (point.x, point.y, point.z)]
        print(flattened_point_cloud)
        # Replace inf and nan values
        processed_point_cloud = []
        for value in flattened_point_cloud:
            if np.isnan(value):
                processed_point_cloud.append(0.0)
            elif np.isinf(value):
                if value > 0:
                    processed_point_cloud.append(100.0)
                else:
                    processed_point_cloud.append(0.0)
            else:
                processed_point_cloud.append(value)

        # Limit the size to 360 values
        processed_point_cloud = processed_point_cloud[:360]

        # Pad the flattened point cloud if it's shorter than the expected size
        if len(processed_point_cloud) < 360:
            processed_point_cloud.extend([0.0] * (360 - len(processed_point_cloud)))

        # Combine amount_explored and processed_point_cloud into a single numpy array
        observation = np.array([amount_explored] + processed_point_cloud, dtype=np.float32)
        return observation