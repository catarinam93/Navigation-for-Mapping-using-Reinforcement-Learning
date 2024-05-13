from controller import Lidar, Compass, GPS
from stable_baselines3 import PPO
from environment import *
from map import *

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

# ----------------------------------- ALGORITHMS AND THEIR HYPERPARAMETERS ---------------------------------------------

    gym.register(id='CustomEnv-ppo', entry_point=lambda: Environment(supervisor))
    env = gym.make('CustomEnv-ppo')

    # Train the model
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("ppo")


def contains_nan(values):
    return any(math.isnan(value) for value in values)

if __name__ == "__main__":
    main()
