from controller import Supervisor, Lidar, Compass, GPS
import math
from map import *
from environment import *
from controllers.transformations import create_tf_matrix
from q_learning import *


def main():

    supervisor: Supervisor = Supervisor()
    timestep: int = 100  # in ms
    map_origin = (0.0, 0.0)
    map_dimensions = (200, 200)
    map_resolution = 0.01

    map: DeterministicOccupancyGrid = DeterministicOccupancyGrid(map_origin, map_dimensions, map_resolution)
    lidar: Lidar = supervisor.getDevice('lidar')
    lidar.enable(timestep)
    lidar.enablePointCloud()

    compass: Compass = supervisor.getDevice('compass')
    compass.enable(timestep)

    gps: GPS = supervisor.getDevice('gps')
    gps.enable(timestep)

    gps_readings: [float] = gps.getValues()
    compass_readings: [float] = compass.getValues()
    # Read the robot's pose
    while contains_nan(gps_readings) or contains_nan(compass_readings):
        gps_readings: [float] = gps.getValues()
        compass_readings: [float] = compass.getValues()
        supervisor.step()

    print("gps:", gps_readings)
    print("compass:", compass_readings)
    supervisor_position: (float, float) = (gps_readings[0], gps_readings[1])
    supervisor_orientation: float = math.atan2(compass_readings[0], compass_readings[1])
    print("robot_position:", supervisor_position)
    print("robot_orientation:", supervisor_orientation)
    supervisor_tf: np.ndarray = create_tf_matrix((supervisor_position[0], supervisor_position[1], 0.0), supervisor_orientation)

    # Read the LiDAR and update the map
    point_cloud = lidar.getPointCloud()
    valid_points = [point for point in point_cloud if
                    not (math.isnan(point.x) or math.isnan(point.y) or math.isnan(point.z))]
    map.update_map(supervisor_tf, valid_points)


# ------------------------------------------------ ENVIRONMENT ---------------------------------------------------------
    env = Environment(supervisor)

# ----------------------------------- ALGORITHMS AND THEIR HYPERPARAMETERS ---------------------------------------------
    learning_rate = 0.8
    discount_factor = 0.95
    exploration_prob = 0.2
    epochs = 1000
    epsilon = 0.01


    # Initialize Q-learning algorithm
    q_learning_agent = QLearning(env, learning_rate, discount_factor, exploration_prob, epochs, epsilon)

    # Train the Q-learning algorithm
    q_learning_agent.learn()


def contains_nan(values):
    return any(math.isnan(value) for value in values)

if __name__ == "__main__":
    main()