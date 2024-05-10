from controller import Robot, Lidar, Compass, GPS
import math
from map import *
from environment import *
from controllers.transformations import create_tf_matrix
from q_learning import *


def main():
    robot: Robot = Robot()
    timestep: int = 100  # in ms

    map_origin = (0.0, 0.0)
    map_dimensions = (200, 200)
    map_resolution = 0.01

    map: DeterministicOccupancyGrid = DeterministicOccupancyGrid(map_origin, map_dimensions, map_resolution)

    lidar: Lidar = robot.getDevice('lidar')
    lidar.enable(timestep)
    lidar.enablePointCloud()

    compass: Compass = robot.getDevice('compass')
    compass.enable(timestep)

    gps: GPS = robot.getDevice('gps')
    gps.enable(timestep)

    # Read the robot's pose
    gps_readings: [float] = gps.getValues()
    robot_position: (float, float) = (gps_readings[0], gps_readings[1])
    compass_readings: [float] = compass.getValues()
    robot_orientation: float = math.atan2(compass_readings[0], compass_readings[1])
    robot_tf: np.ndarray = create_tf_matrix((robot_position[0], robot_position[1], 0.0), robot_orientation)

    # Read the LiDAR and update the map
    map.update_map(robot_tf, lidar.getPointCloud())

# ------------------------------------------------ ENVIRONMENT ---------------------------------------------------------
    env = Environment(robot)

# ----------------------------------- ALGORITHMS AND THEIR HYPERPARAMETERS ---------------------------------------------
    learning_rate = 0.8
    discount_factor = 0.95
    exploration_prob = 0.2
    epochs = 1000
    epsilon = 0.01

    QLearning(env, learning_rate, discount_factor, exploration_prob, epochs, epsilon)