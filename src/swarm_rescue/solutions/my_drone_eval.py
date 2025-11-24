import math
import numpy as np
import cv2
from collections import deque

from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.utils.grid import Grid
from swarm_rescue.simulation.utils.pose import Pose
from swarm_rescue.simulation.utils.constants import MAX_RANGE_LIDAR_SENSOR
from swarm_rescue.simulation.utils.utils import normalize_angle

class OccupancyGrid(Grid):
    """
    Occupancy grid that updates map based on Lidar data.
    """
    def __init__(self, size_area_world, resolution: float, lidar):
        super().__init__(size_area_world=size_area_world, resolution=resolution)
        self.lidar = lidar
        self.grid = np.zeros((self.x_max_grid, self.y_max_grid))

    def update_grid(self, pose: Pose):
        EVERY_N = 2
        EMPTY_ZONE_VALUE = -0.4
        OBSTACLE_ZONE_VALUE = 1.0
        FREE_ZONE_VALUE = -2.0
        THRESHOLD_MIN = -20
        THRESHOLD_MAX = 20

        lidar_values = self.lidar.get_sensor_values()
        if lidar_values is None: return

        lidar_dist = lidar_values[::EVERY_N].copy()
        lidar_angles = self.lidar.ray_angles[::EVERY_N].copy()

        cos_rays = np.cos(lidar_angles + pose.orientation)
        sin_rays = np.sin(lidar_angles + pose.orientation)

        valid_range_threshold = MAX_RANGE_LIDAR_SENSOR * 0.95

        lidar_dist_empty = np.minimum(lidar_dist, valid_range_threshold)
        points_x = pose.position[0] + np.multiply(lidar_dist_empty, cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist_empty, sin_rays)

        for pt_x, pt_y in zip(points_x, points_y):
            self.add_value_along_line(pose.position[0], pose.position[1],
                                      pt_x, pt_y, EMPTY_ZONE_VALUE)

        select_collision = lidar_dist < valid_range_threshold
        points_x = pose.position[0] + np.multiply(lidar_dist, cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist, sin_rays)
        
        self.add_points(points_x[select_collision], points_y[select_collision], OBSTACLE_ZONE_VALUE)
        self.add_points(pose.position[0], pose.position[1], FREE_ZONE_VALUE)
        self.grid = np.clip(self.grid, THRESHOLD_MIN, THRESHOLD_MAX)

    def display(self, pose: Pose, history, debug_text, wall_dists, name="Mapping"):
        img = self.grid.T
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)

        if len(history) > 1:
            pts = []
            for pos in history:
                px, py = self._conv_world_to_grid(pos[0], pos[1])
                pts.append([px, py])
            pts = np.array(pts, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img_color, [pts], False, (255, 255, 150), 1)

        pt_x, pt_y = self._conv_world_to_grid(pose.position[0], pose.position[1])
        if 0 <= pt_x < self.x_max_grid and 0 <= pt_y < self.y_max_grid:
            radius_follow = int(wall_dists[0] / self.resolution)
            radius_engage = int(wall_dists[1] / self.resolution)
            cv2.circle(img_color, (int(pt_x), int(pt_y)), radius_engage, (0, 0, 255), 1)
            cv2.circle(img_color, (int(pt_x), int(pt_y)), radius_follow, (0, 255, 0), 1)
            cv2.circle(img_color, (int(pt_x), int(pt_y)), 5, (255, 255, 255), -1)
            end_x = pt_x + 10 * np.cos(pose.orientation)
            end_y = pt_y - 10 * np.sin(pose.orientation)
            cv2.line(img_color, (int(pt_x), int(pt_y)), (int(end_x), int(end_y)), (0, 0, 255), 2)

        display_img = cv2.resize(img_color, (800, 600), interpolation=cv2.INTER_NEAREST)
        cv2.imshow(name, display_img)
        cv2.waitKey(1)


class MyDroneEval(DroneAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grid = OccupancyGrid(size_area_world=self.size_area, resolution=5.0, lidar=self.lidar())
        self.pose = Pose()
        self.use_odometry = False 

        self.CRUISING_SPEED = 0.6
        self.ROTATION_SPEED = 0.4
        self.WALL_FOLLOW_DIST = 90.0
        self.WALL_ENGAGE_DIST = 110.0

        self.angleStopTurning = 0
        self.isTurning = False
        self.position_history = deque(maxlen=100)
        self.stuck_mode = False
        self.stuck_timer = 0
        self.debug_info = ""

    def define_message_for_all(self):
        pass

    def update_pose(self):
        gps_pos = self.measured_gps_position()
        compass_angle = self.measured_compass_angle()
        gps_valid = gps_pos is not None and not np.isnan(gps_pos).any()
        compass_valid = compass_angle is not None and not np.isnan(compass_angle)

        if gps_valid and compass_valid:
            self.pose.position = gps_pos
            self.pose.orientation = compass_angle
            self.use_odometry = False
        else:
            self.use_odometry = True
            odom = self.odometer_values()
            if odom is not None:
                dist, alpha, theta = odom
                self.pose.orientation = normalize_angle(self.pose.orientation + theta)
                move_angle = self.pose.orientation - theta + alpha
                dx = dist * math.cos(move_angle)
                dy = dist * math.sin(move_angle)
                self.pose.position += np.array([dx, dy])

    def check_if_stuck(self):
        self.position_history.append(self.pose.position.copy())
        if len(self.position_history) < 100: return False
        points = np.array(self.position_history)
        dist = np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0))
        return dist < 50.0

    def display_lidar(self):
        if self.identifier != 0: return
        lidar_values = self.lidar_values()
        if lidar_values is None: return
        lidar_angles = self.lidar().ray_angles
        img_size = 400
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        center = (img_size // 2, img_size // 2)
        scale = img_size / (2 * MAX_RANGE_LIDAR_SENSOR)
        valid_threshold = MAX_RANGE_LIDAR_SENSOR * 0.95

        for dist, angle in zip(lidar_values, lidar_angles):
            if dist >= valid_threshold: continue
            x = center[0] - int(dist * math.sin(angle) * scale)
            y = center[1] - int(dist * math.cos(angle) * scale)
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        cv2.circle(img, center, 5, (0, 0, 255), -1)
        cv2.line(img, center, (center[0], center[1] - 30), (0, 0, 255), 2)
        cv2.imshow("Lidar Scan", img)
        cv2.waitKey(1)

    def robust_wall_follow_logic(self):
        """
        Calculates a target vector that aligns parallel to the nearest wall.
        """
        lidar_values = self.lidar_values()
        lidar_angles = self.lidar().ray_angles
        if lidar_values is None: return {"forward": self.CRUISING_SPEED, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        # Filter valid points
        valid_mask = lidar_values < self.WALL_ENGAGE_DIST * 1.2
        if not np.any(valid_mask):
            # No wall nearby -> Forward
            return {"forward": self.CRUISING_SPEED, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        relevant_dists = lidar_values[valid_mask]
        relevant_angles = lidar_angles[valid_mask]

        # 1. Find the closest point (The "Wall Anchor")
        min_idx = np.argmin(relevant_dists)
        min_dist = relevant_dists[min_idx]
        min_angle = relevant_angles[min_idx] # Angle to the closest wall point relative to drone heading

        # 2. Decide Turn Direction
        # If wall is on Left (angle > 0), we want to align such that wall stays on left.
        # Tangent is Wall_Angle - 90 deg.
        # If wall is on Right (angle < 0), Tangent is Wall_Angle + 90 deg.
        
        if min_angle > 0: # Wall is on LEFT
            target_angle = min_angle - (math.pi / 2)
        else: # Wall is on RIGHT
            target_angle = min_angle + (math.pi / 2)

        # 3. Distance Correction (The "Magnetic" Pull/Push)
        # We want distance to be WALL_FOLLOW_DIST.
        # error positive = too far (need to turn towards wall)
        # error negative = too close (need to turn away from wall)
        dist_error = min_dist - self.WALL_FOLLOW_DIST
        
        # Correction angle: Turn towards wall if far, away if close.
        # Max correction is limited to avoid 90-degree slams
        correction = math.radians(30) * np.clip(dist_error / 50.0, -1.0, 1.0)
        
        # If wall is on LEFT: Far (+error) -> Need to turn Left (+angle).
        # If wall is on RIGHT: Far (+error) -> Need to turn Right (-angle).
        if min_angle > 0: # Left Wall
            target_angle += correction
        else: # Right Wall
            target_angle -= correction

        # 4. Calculate Rotation Command
        # Target angle is relative to current heading. We want to reduce this to 0.
        # Normalize to -pi to pi
        target_angle = normalize_angle(target_angle)
        
        # Proportional controller for rotation
        rot_cmd = np.clip(target_angle * 2.0, -1.0, 1.0) * self.ROTATION_SPEED

        # 5. Speed Control (Slow down in corners)
        # Check directly front (-30 to 30 deg)
        front_mask = (relevant_angles > -0.5) & (relevant_angles < 0.5)
        if np.any(front_mask):
            front_min = np.min(relevant_dists[front_mask])
            speed_factor = max(0.0, (front_min - 40) / 60)
        else:
            speed_factor = 1.0
            
        fwd_cmd = self.CRUISING_SPEED * speed_factor

        return {"forward": fwd_cmd, "lateral": 0.0, "rotation": rot_cmd, "grasper": 0}

    def safe_limit_control(self, cmd):
        cmd["forward"] = max(-0.5, min(0.8, cmd["forward"]))
        cmd["lateral"] = 0.0
        cmd["rotation"] = max(-0.8, min(0.8, cmd["rotation"]))
        return cmd

    def control(self) -> CommandsDict:
        self.update_pose()
        self.grid.update_grid(self.pose)
        
        if self.identifier == 0:
            title = "Mapping (GPS)" if not self.use_odometry else "Mapping (NO GPS)"
            dists = (self.WALL_FOLLOW_DIST, self.WALL_ENGAGE_DIST)
            self.grid.display(self.pose, self.position_history, self.debug_info, dists, name=title)
            self.display_lidar()

        # 1. Stuck Recovery
        if self.stuck_mode:
            self.stuck_timer -= 1
            if self.stuck_timer <= 0: self.stuck_mode = False
            return self.safe_limit_control({"forward": 0.5, "lateral": 0.0, "rotation": 0.5, "grasper": 0})

        if self.check_if_stuck():
            self.stuck_mode = True
            self.stuck_timer = 50 
            self.position_history.clear()
            return self.safe_limit_control({"forward": -0.5, "lateral": 0.0, "rotation": 1.0, "grasper": 0})

        # 2. Robust Wall Follower
        cmd = self.robust_wall_follow_logic()
        
        return self.safe_limit_control(cmd)