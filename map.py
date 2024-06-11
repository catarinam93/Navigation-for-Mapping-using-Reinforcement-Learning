'''This script defines a class DeterministicOccupancyGrid that represents a deterministic occupancy grid map. 
The occupancy grid map is a common representation used in robotics to model the environment, where each 
cell of the grid represents the probability of occupancy. The class provides methods to initialize the grid, 
update it based on sensor readings, and query its state. Additionally, utility functions for geometric calculations 
and collision detection are included, offering comprehensive support for navigation and environment perception tasks 
in robot control applications.
The code adapted from the one provided by the professor Gonçalo Leão'''

from controllers.utils import bresenham
from occupancy_grid import OccupancyGrid
import numpy as np
from controllers.transformations import get_translation
from typing import List, Tuple
import matplotlib.pyplot as plt

class DeterministicOccupancyGrid(OccupancyGrid):
    def __init__(self, origin: (float, float), dimensions: (int, int), resolution: float):
        # Initialize the superclass OccupancyGrid
        super().__init__(origin, dimensions, resolution)
        # Initialize the occupancy grid with unknown values (0.5)
        self.occupancy_grid: np.ndarray = np.full((100, 100), 0.5, dtype=np.float32)
        # List of pixels that represent walls
        self.wall_pixels: List[Tuple[int, int]] = []

    def is_wall(self, coords: Tuple[int, int]) -> bool:
        """
        Checks if a given position contains a wall.
        :param coords: The coordinates (x, y) of the pixel to be checked.
        :return: True if the position contains a wall, False otherwise.
        """
        return coords in self.wall_pixels

    def update_map(self, robot_tf: np.ndarray, lidar_points: [(float, float)]):
        # Get the grid coordinate for the robot's pose
        robot_coord: (int, int) = self.real_to_grid_coords(get_translation(robot_tf)[0:2])

        # Get the grid coordinates for the lidar points
        grid_lidar_coords: [(int, int)] = []
        for x, y in lidar_points:
            # Transform lidar points to grid coordinates using the robot's transformation matrix
            coord: (int, int) = self.real_to_grid_coords(np.dot(robot_tf, [x, y, 0.0, 1.0])[0:2])
            grid_lidar_coords.append(coord)

        # Mark the robot's position as free
        self.update_cell(robot_coord, False)

        # Mark the cells leading up to the lidar points as free
        for coord in grid_lidar_coords:
            # Use Bresenham's algorithm to find intermediate points between the robot and lidar points
            for mid_coord in bresenham(robot_coord, coord)[1:-1]:
                self.update_cell(mid_coord, False)

        # Mark the cells at the lidar points as occupied
        for coord in grid_lidar_coords:
            self.update_cell(coord, True)

        explored_cells = set(grid_lidar_coords)

        return len(explored_cells)  # Return the number of explored cells

    def update_cell(self, coords: (int, int), is_occupied: bool):
        if self.are_grid_coords_in_bounds(coords):
            # Check if the cell has already been explored
            if self.occupancy_grid[coords] != 0.5:  # Cell is not unknown
                return True  # Cell has already been explored

            # Update the grid cell
            self.occupancy_grid[coords] = 1 if is_occupied else 0

        return False  # Cell was not previously explored

    def percentage_explored(self) -> float:
        # Calculate the total number of cells
        total_cells = np.size(self.occupancy_grid)
        # Calculate the number of explored cells
        explored_cells = np.count_nonzero(self.occupancy_grid != 0.5)
        # Calculate the percentage of explored cells
        percentage = (explored_cells / total_cells) * 100
        return percentage

    def all_cells_explored(self) -> bool:
        # Calculate the total number of cells
        total_cells = np.size(self.occupancy_grid)
        # Calculate the number of explored cells
        explored_cells = np.count_nonzero(self.occupancy_grid != 0.5)

        # Check if all cells have been explored
        return explored_cells == total_cells

    def get_flattened_state(self) -> np.ndarray:
        """
        Returns the state of the map as a one-dimensional array.
        Unknown cells are represented by 0.5, free cells by 0, and occupied cells by 1.
        """
        return self.occupancy_grid.flatten()

    def plot_grid(self, save_path=None):
        plt.figure()  # New figure
        plt.imshow(self.occupancy_grid, cmap='gray', origin='lower')
        plt.colorbar(label='Occupancy Probability')
        plt.title('Deterministic Occupancy Grid')
        plt.xlabel('X')
        plt.ylabel('Y')
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
