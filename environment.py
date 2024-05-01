import gymnasium as gym
from map import *
from controller import Supervisor, Field
from typing import List

class Environment(gym.Env):
    def __init__(self):
        self.map = None
        self.timesteps = 0

        self.sensor_bounds = [0, 255]  # Sensor bounds of the environment
        '''self.angle_bounds = [-np.pi, np.pi]  # Angle bounds of the environment
        self.distance_bounds = [0, np.inf]  # Distance bounds of the environment'''

        self.angular_vel_bounds = [-1.0, 1.0]  # Angular Velocity bounds of the environment
        self.linear_vel_bounds = [-1.0, 1.0]  # Linear Velocity bounds of the environment

        self.action_space = gym.spaces.Box(low=np.array([self.linear_vel_bounds[0], self.angular_vel_bounds[0]]),
                                            high=np.array([self.linear_vel_bounds[1], self.angular_vel_bounds[1]]),
                                            dtype=np.float32)  # Action space of the environment

        self.observation_space = gym.spaces.Box(low=self.sensor_bounds[0],
                                                  high=self.sensor_bounds[1],
                                                  shape=(100), #numero de raios
                                                  dtype=np.float32)

    def warp_robot(supervisor: Supervisor, robot_def_name: str, new_position: (float, float)) -> None:
        robot_node = supervisor.getFromDef(robot_def_name)
        trans_field: Field = robot_node.getField("translation")
        translation: List[float] = [new_position[0], new_position[1], 0]
        trans_field.setSFVec3f(translation)
        robot_node.resetPhysics()

    # Reset the environment
    def reset(self):
        super().reset()  # Reset the environment

        self.map = DeterministicOccupancyGrid(OccupancyGrid) # Resets the map

        self.warp_robot() # fazer reset à orientação do utils

        # Reset all the variables
        self.timesteps = 0  # Reset the timesteps

        observation = self._get_obs()
        # info = self._get_info()
        return observation  # , info

    def terminated(self):
        #completar
        return false

    def calculate_reward(self):
        #completar
        reward = 0
        return reward

    '''def _get_obs(self):
        # Calculate the percentage of map explored
        total_cells = np.prod(self.map.occupancy_grid.shape)
        explored_cells = np.count_nonzero(self.map.occupancy_grid != 0.5) 
        percentage_explored = (explored_cells / total_cells) * 100

        observation = {"Percentage of map explored": percentage_explored}
        return observation'''

    def _get_obs(self):
        amount = 0
        return {"Amount of map mapped", amount}

    # Step the environment
    def step(self, action):
        # An episode is done if the robot has completed the map

        # Reward
        reward = self.calculate_reward()

        terminated = self.terminated()

        observation = self._get_obs()

        return observation, reward, terminated  # , info

