import math
import numpy as np
import cv2
from typing import Optional, Dict

from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.utils.grid import Grid
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.simulation.utils.pose import Pose
from swarm_rescue.simulation.utils.utils import normalize_angle, clamp
from swarm_rescue.simulation.utils.constants import MAX_RANGE_LIDAR_SENSOR


class OccupancyGrid(Grid):
    """
    A grid-based map that accumulates sensor data to build a representation 
    of the environment (free space vs obstacles).
    """

    def __init__(self, size_area_world, resolution: float, lidar):
        # Initialize the parent Grid class which handles coordinate conversion and storage
        super().__init__(size_area_world=size_area_world, resolution=resolution)
        self.lidar = lidar
        # Initialize zoomed grid for better visualization
        self.zoomed_grid = np.zeros((self.x_max_grid, self.y_max_grid))

    def update_grid(self, pose: Pose):
        """
        Updates the grid based on the current Lidar observation and drone pose.
        """
        # Mapping parameters
        EVERY_N = 3                 # Skip rays to improve performance
        LIDAR_DIST_CLIP = 40.0      # Safety margin for empty space detection
        EMPTY_ZONE_VALUE = -0.602   # Value to decrease probability of obstacle
        OBSTACLE_ZONE_VALUE = 2.0   # Value to increase probability of obstacle
        FREE_ZONE_VALUE = -4.0      # Strong value for drone's current position
        THRESHOLD_MIN = -40         # Min saturation value for grid cells
        THRESHOLD_MAX = 40          # Max saturation value for grid cells

        # Retrieve sensor data
        lidar_values = self.lidar.get_sensor_values()
        if lidar_values is None:
            return

        # Downsample data for speed
        lidar_dist = lidar_values[::EVERY_N].copy()
        lidar_angles = self.lidar.ray_angles[::EVERY_N].copy()

        # Calculate the absolute angle of each ray in the world frame
        # (Ray Angle relative to drone + Drone Orientation)
        cos_rays = np.cos(lidar_angles + pose.orientation)
        sin_rays = np.sin(lidar_angles + pose.orientation)

        # Define valid range for mapping (slightly less than sensor max to avoid noise)
        max_usable_range = MAX_RANGE_LIDAR_SENSOR * 0.9

        # --- 1. Update Free Space (Ray Tracing) ---
        # We consider the space up to the hit point (minus a clip margin) as free.
        lidar_dist_empty = np.maximum(lidar_dist - LIDAR_DIST_CLIP, 0.0)
        lidar_dist_empty_clip = np.minimum(lidar_dist_empty, max_usable_range)
        
        points_x = pose.position[0] + np.multiply(lidar_dist_empty_clip, cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist_empty_clip, sin_rays)

        # Trace lines from drone position to calculated points clearing the grid along the way
        for pt_x, pt_y in zip(points_x, points_y):
            self.add_value_along_line(pose.position[0], pose.position[1],
                                      pt_x, pt_y,
                                      EMPTY_ZONE_VALUE)

        # --- 2. Update Obstacles ---
        # Only mark obstacles if they are within reliable range
        select_collision = lidar_dist < max_usable_range
        
        obs_x = pose.position[0] + np.multiply(lidar_dist, cos_rays)
        obs_y = pose.position[1] + np.multiply(lidar_dist, sin_rays)

        # Filter only valid collision points
        obs_x = obs_x[select_collision]
        obs_y = obs_y[select_collision]

        self.add_points(obs_x, obs_y, OBSTACLE_ZONE_VALUE)

        # --- 3. Mark Drone Position ---
        # The drone's current center is definitely free space
        self.add_points(pose.position[0], pose.position[1], FREE_ZONE_VALUE)

        # --- 4. Clamp Values ---
        # Keep grid values within bounds to prevent overflow/underflow
        self.grid = np.clip(self.grid, THRESHOLD_MIN, THRESHOLD_MAX)

        # --- 5. Update zoomed grid for visualization ---
        # We create a smaller version of the grid for easier viewing
        self.zoomed_grid = self.grid.copy()
        # Resize for better visibility (50% scale)
        new_zoomed_size = (int(self.size_area_world[1] * 0.5),
                           int(self.size_area_world[0] * 0.5))
        self.zoomed_grid = cv2.resize(self.zoomed_grid, new_zoomed_size,
                                      interpolation=cv2.INTER_NEAREST)


class MyDroneEval(DroneAbstract):
    """
    Evaluation Drone class for single-drone mapping mission.
    """

    def __init__(self, identifier: Optional[int] = None, misc_data: Optional[MiscData] = None, **kwargs):
        # display_lidar_graph=False to hide the line plot graph
        super().__init__(identifier=identifier, misc_data=misc_data, display_lidar_graph=False, **kwargs)
        
        # Initialize the Occupancy Grid
        # Resolution: 10 pixels per grid cell (adjust for coarser/finer map)
        resolution = 10 
        self.grid = OccupancyGrid(size_area_world=self.size_area, 
                                  resolution=resolution, 
                                  lidar=self.lidar())
        
        # Counter to limit how often we redraw the map (for performance)
        self.iteration = 0

        # --- PID Variables ---
        self.prev_rotation_error = 0.0

    def define_message_for_all(self):
        """
        No communication needed for single drone mission.
        """
        pass
        
    def display_local_lidar(self):
        """
        Visualizes the raw lidar data on a black background.
        Drone is in the center, obstacles are red dots.
        """
        lidar_values = self.lidar().get_sensor_values()
        if lidar_values is None:
            return

        ray_angles = self.lidar().ray_angles

        # Create a black image (500x500 pixels)
        img_size = 500
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        center = img_size // 2
        
        # Draw Drone in the center (Blue circle)
        # BGR color: (255, 0, 0) is Blue
        cv2.circle(img, (center, center), 5, (255, 0, 0), -1)

        # Filter valid points (ignore max range which usually means no obstacle)
        valid_points = lidar_values < (MAX_RANGE_LIDAR_SENSOR * 0.99)
        
        dists = lidar_values[valid_points]
        angles = ray_angles[valid_points]

        # Convert Polar (distance, angle) to Cartesian (x, y) for image
        # We flip the Y axis because in images, Y increases downwards
        xs = center + (dists * np.cos(angles)).astype(int)
        ys = center - (dists * np.sin(angles)).astype(int)

        # Draw red dots for obstacles
        for x, y in zip(xs, ys):
            # Ensure points are within image bounds
            if 0 <= x < img_size and 0 <= y < img_size:
                # BGR color: (0, 0, 255) is Red
                cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

        cv2.imshow("Local Lidar View", img)
        cv2.waitKey(1)

    def control(self) -> CommandsDict:
        """
        Main control loop called at every timestep.
        1. Reads sensors (GPS, Compass, Lidar).
        2. Updates the internal map.
        3. Computes movement commands to explore safely.
        """
        
        # --- 1. Mapping Step ---
        # Get current estimated position from noisy GPS/Compass
        # Note: In a real scenario with NO_GPS_ZONE, you would integrate the Odometer here.
        gps_pos = self.measured_gps_position()
        compass_angle = self.measured_compass_angle()
        
        # Only update map if we have valid position data
        if gps_pos is not None and compass_angle is not None:
            current_pose = Pose(np.asarray(gps_pos), compass_angle)
            self.grid.update_grid(pose=current_pose)

            # Display logic
            self.iteration += 1
            if self.iteration % 5 == 0:
                # --- UNCOMMENTED OCCUPANCY GRID DISPLAY ---
                self.grid.display(self.grid.grid, current_pose, title="Occupancy Grid")
                self.grid.display(self.grid.zoomed_grid, current_pose, title="Zoomed Occupancy Grid")
                
                # --- NEW LOCAL LIDAR DISPLAY ---
                self.display_local_lidar()

        # --- 2. Navigation Step ---
        # Compute movement based on Lidar data
        command = self.process_lidar_sensor()

        return command

    def process_lidar_sensor(self) -> CommandsDict:
        """
        Analyzes Lidar data to determine safe movement.
        Uses PD Control for smooth rotation.
        """
        command = {"forward": 1.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
        
        lidar_values = self.lidar().get_sensor_values()
        
        # If sensor is disabled (e.g. broken or special zone), stay still to be safe
        if lidar_values is None:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        ray_angles = self.lidar().ray_angles
        
        # --- Analysis ---
        # 1. Find direction of maximum free space (Target Heading)
        far_angle_idx = np.argmax(lidar_values)
        far_angle = ray_angles[far_angle_idx]
        
        # 2. Find distance and direction of nearest obstacle
        min_dist = min(lidar_values)