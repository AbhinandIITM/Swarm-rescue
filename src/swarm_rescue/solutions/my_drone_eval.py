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
    def __init__(self, size_area_world, resolution: float, lidar):
        super().__init__(size_area_world=size_area_world, resolution=resolution)
        self.lidar = lidar
        self.zoomed_grid = np.zeros((self.x_max_grid, self.y_max_grid))

    def update_grid(self, pose: Pose):
        EVERY_N = 3
        LIDAR_DIST_CLIP = 40.0
        EMPTY_ZONE_VALUE = -0.602
        OBSTACLE_ZONE_VALUE = 2.0
        FREE_ZONE_VALUE = -4.0
        THRESHOLD_MIN = -40
        THRESHOLD_MAX = 40

        lidar_values = self.lidar.get_sensor_values()
        if lidar_values is None:
            return

        lidar_dist = lidar_values[::EVERY_N].copy()
        lidar_angles = self.lidar.ray_angles[::EVERY_N].copy()
        cos_rays = np.cos(lidar_angles + pose.orientation)
        sin_rays = np.sin(lidar_angles + pose.orientation)
        max_usable_range = MAX_RANGE_LIDAR_SENSOR * 0.9

        lidar_dist_empty = np.maximum(lidar_dist - LIDAR_DIST_CLIP, 0.0)
        lidar_dist_empty_clip = np.minimum(lidar_dist_empty, max_usable_range)
        
        points_x = pose.position[0] + np.multiply(lidar_dist_empty_clip, cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist_empty_clip, sin_rays)

        for pt_x, pt_y in zip(points_x, points_y):
            self.add_value_along_line(pose.position[0], pose.position[1], pt_x, pt_y, EMPTY_ZONE_VALUE)

        select_collision = lidar_dist < max_usable_range
        obs_x = pose.position[0] + np.multiply(lidar_dist, cos_rays)
        obs_y = pose.position[1] + np.multiply(lidar_dist, sin_rays)
        obs_x = obs_x[select_collision]
        obs_y = obs_y[select_collision]

        self.add_points(obs_x, obs_y, OBSTACLE_ZONE_VALUE)
        self.add_points(pose.position[0], pose.position[1], FREE_ZONE_VALUE)
        self.grid = np.clip(self.grid, THRESHOLD_MIN, THRESHOLD_MAX)
        
        self.zoomed_grid = self.grid.copy()
        new_zoomed_size = (int(self.size_area_world[1] * 0.5), int(self.size_area_world[0] * 0.5))
        self.zoomed_grid = cv2.resize(self.zoomed_grid, new_zoomed_size, interpolation=cv2.INTER_NEAREST)


class MyDroneEval(DroneAbstract):
    def __init__(self, identifier: Optional[int] = None, misc_data: Optional[MiscData] = None, **kwargs):
        super().__init__(identifier=identifier, misc_data=misc_data, display_lidar_graph=False, **kwargs)
        resolution = 10 
        self.grid = OccupancyGrid(size_area_world=self.size_area, resolution=resolution, lidar=self.lidar())
        self.iteration = 0
        self.prev_rotation_error = 0.0

    def define_message_for_all(self):
        pass
        
    def display_local_lidar(self):
        lidar_values = self.lidar().get_sensor_values()
        if lidar_values is None: return

        ray_angles = self.lidar().ray_angles
        img_size = 500
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        center = img_size // 2
        cv2.circle(img, (center, center), 5, (255, 0, 0), -1)

        valid_points = lidar_values < (MAX_RANGE_LIDAR_SENSOR * 0.99)
        dists = lidar_values[valid_points]
        angles = ray_angles[valid_points]

        xs = center + (dists * np.cos(angles)).astype(int)
        ys = center - (dists * np.sin(angles)).astype(int)

        for x, y in zip(xs, ys):
            if 0 <= x < img_size and 0 <= y < img_size:
                cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
        cv2.imshow("Local Lidar View", img)
        cv2.waitKey(1)

    def control(self) -> CommandsDict:
        gps_pos = self.measured_gps_position()
        compass_angle = self.measured_compass_angle()
        
        if gps_pos is not None and compass_angle is not None:
            current_pose = Pose(np.asarray(gps_pos), compass_angle)
            self.grid.update_grid(pose=current_pose)
            self.iteration += 1
            if self.iteration % 5 == 0:
                try:
                    self.grid.display(self.grid.grid, current_pose, title="Occupancy Grid")
                    self.grid.display(self.grid.zoomed_grid, current_pose, title="Zoomed Occupancy Grid")
                    self.display_local_lidar()
                except:
                    pass

        return self.process_lidar_sensor()

    def process_lidar_sensor(self) -> CommandsDict:
        command = {"forward": 1.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
        lidar_values = self.lidar().get_sensor_values()
        
        if lidar_values is None:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        ray_angles = self.lidar().ray_angles
        far_angle_idx = np.argmax(lidar_values)
        far_angle = ray_angles[far_angle_idx]
        
        # DEBUG PRINT: Check if drone sees obstacles
        min_dist = min(lidar_values)
        # print(f"Min Dist: {min_dist:.1f}, Far Angle: {far_angle:.2f}")

        # --- PD Control for Rotation ---
        error = normalize_angle(far_angle)
        Kp = 1.5
        Kd = 0.5
        derivative = error - self.prev_rotation_error
        self.prev_rotation_error = error
        rotation_command = (Kp * error) + (Kd * derivative)
        command["rotation"] = clamp(rotation_command, -1.0, 1.0)

        # --- Smoother Avoidance ---
        # Critical stop only if VERY close (20 pixels)
        if min_dist < 20: 
            command["forward"] = -0.2 # Back up slightly
            near_angle_idx = np.argmin(lidar_values)
            near_angle = ray_angles[near_angle_idx]
            command["rotation"] = -1.0 if near_angle > 0 else 1.0
            
        # Slow down if somewhat close (60 pixels) but keep moving
        elif min_dist < 60:
            command["forward"] = 0.3
        
        return command