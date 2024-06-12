'''This script calculates the average number of timesteps for different methods (PPO, A2C, SAC, TD3) across multiple maps.'''

import os
import re
import numpy as np

# This function extracts the number of timesteps from a filename
def extract_timesteps_from_filename(filename):
    match = re.search(r'(\d+)_timesteps\.png', filename)
    if match:
        return int(match.group(1))
    return None

# This function calculates the average timesteps in a folder
def calculate_average_timesteps_in_folder(folder_path):
    timesteps = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.png'):
            timesteps.append(extract_timesteps_from_filename(file_name))
    return np.mean(timesteps)

# Define the paths to your image directories
base_path = 'maps_images'

# Calculate the average timesteps for the PPO method in each map
for i in range(5):
    map_folder = os.path.join(base_path, f'PPO/map{i}')
    avg_timesteps = calculate_average_timesteps_in_folder(map_folder)
    print(f"Average timesteps for the PPO method in map {i}: {avg_timesteps}")

# Calculate the average timesteps for the A2C method in each map
for i in range(5):
    map_folder = os.path.join(base_path, f'A2C/map{i}')
    avg_timesteps = calculate_average_timesteps_in_folder(map_folder)
    print(f"Average timesteps for the A2C method in map {i}: {avg_timesteps}")

# Calculate the average timesteps for the SAC method in each map
map_folder = os.path.join(base_path, f'SAC/map0')
avg_timesteps = calculate_average_timesteps_in_folder(map_folder)
print(f"Average timesteps for the SAC method in map 0: {avg_timesteps}")

# Calculate the average timesteps for the TD3 method in each map
map_folder = os.path.join(base_path, f'TD3/map0')
avg_timesteps = calculate_average_timesteps_in_folder(map_folder)
print(f"Average timesteps for the TD3 method in map 0: {avg_timesteps}")