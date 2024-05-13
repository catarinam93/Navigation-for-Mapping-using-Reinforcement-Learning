from controllers.utils import bresenham
from controllers.occupancy_grid import OccupancyGrid
from controller import LidarPoint
import numpy as np
from controllers.transformations import get_translation
import math


class DeterministicOccupancyGrid(OccupancyGrid):
    def __init__(self, origin: (float, float), dimensions: (int, int), resolution: float):
        super().__init__(origin, dimensions, resolution)

        # Initialize the grid
        self.occupancy_grid: np.ndarray = np.full(dimensions, 0.5, dtype=np.float32)

    def update_map(self, robot_tf: np.ndarray, lidar_points: [LidarPoint]):
        # Get the grid coord for the robot pose
        robot_coord: (int, int) = self.real_to_grid_coords(get_translation(robot_tf)[0:2])

        # Get the grid coords for the lidar points
        grid_lidar_coords: [(int, int)] = []
        for point in lidar_points:
            coord: (int, int) = self.real_to_grid_coords(np.dot(robot_tf, [point.x, point.y, 0.0, 1.0])[0:2])
            grid_lidar_coords.append(coord)

        # Set as free the cell of the robot's position
        self.update_cell(robot_coord, False)

        # Set as free the cells leading up to the lidar points
        for coord in grid_lidar_coords:
            for mid_coord in bresenham(robot_coord, coord)[1:-1]:
                self.update_cell(mid_coord, False)

        # Set as occupied the cells for the lidar points
        for coord in grid_lidar_coords:
            self.update_cell(coord, True)

        explored_cells = set(grid_lidar_coords)

        return len(explored_cells)  # retornar numero de celulas exploradas

    def update_cell(self, coords: (int, int), is_occupied: bool):
        if self.are_grid_coords_in_bounds(coords):
            # Check if the cell has already been explored
            if self.occupancy_grid[coords] != 0.5:  # Cell is not unknown
                return True  # Cell has already been explored

            # Update the grid cell
            self.occupancy_grid[coords] = 1 if is_occupied else 0

        return False  # Cell has not been explored before

    def percentage_explored(self) -> float:
        total = len(self.occupancy_grid)
        explored_cells = sum(1 for coord in self.occupancy_grid if coord == 1)
        percentage = (explored_cells / total) * 100
        return percentage

    def all_cells_explored(self) -> bool:
        explored_cells = 0

        for coord in self.occupancy_grid:
            if coord == 1:
                explored_cells += 1

        if self.percentage_explored() >= 0.95:
            return True
        else:
            return False
