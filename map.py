from controllers.utils import bresenham
from controllers.TP4.occupancy_grid import OccupancyGrid
from controller import LidarPoint, Supervisor
import numpy as np
from controllers.transformations import get_translation

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

        return len(grid_lidar_coords) # retornar numero de celulas exploradas

    def update_cell(self, coords: (int, int), is_occupied: bool):
        if self.are_grid_coords_in_bounds(coords):
            # Update the grid cell
            self.occupancy_grid[coords] = 1 if is_occupied else 0

        return # em vez de none se já tinha sido explorado

