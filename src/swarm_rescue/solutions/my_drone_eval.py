import math
import numpy as np
import cv2

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
        # Performance optimization: Process 1 out of every 2 rays
        EVERY_N = 2
        LIDAR_DIST_CLIP = 40.0
        
        # Log-odds values for Bayesian update
        EMPTY_ZONE_VALUE = -0.6
        OBSTACLE_ZONE_VALUE = 2.0
        FREE_ZONE_VALUE = -4.0
        THRESHOLD_MIN = -40
        THRESHOLD_MAX = 40

        lidar_values = self.lidar.get_sensor_values()
        if lidar_values is None:
            return

        lidar_dist = lidar_values[::EVERY_N].copy()
        lidar_angles = self.lidar.ray_angles[::EVERY_N].copy()

        # Compute absolute ray angles
        cos_rays = np.cos(lidar_angles + pose.orientation)
        sin_rays = np.sin(lidar_angles + pose.orientation)

        # Maximum range we trust for mapping
        max_range = MAX_RANGE_LIDAR_SENSOR * 0.9

        # 1. Update Empty Zones
        lidar_dist_empty = np.maximum(lidar_dist - LIDAR_DIST_CLIP, 0.0)
        lidar_dist_empty_clip = np.minimum(lidar_dist_empty, max_range)
        points_x = pose.position[0] + np.multiply(lidar_dist_empty_clip, cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist_empty_clip, sin_rays)

        for pt_x, pt_y in zip(points_x, points_y):
            self.add_value_along_line(pose.position[0], pose.position[1],
                                      pt_x, pt_y, EMPTY_ZONE_VALUE)

        # 2. Update Obstacles
        select_collision = lidar_dist < max_range
        points_x = pose.position[0] + np.multiply(lidar_dist, cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist, sin_rays)
        
        self.add_points(points_x[select_collision], points_y[select_collision], OBSTACLE_ZONE_VALUE)

        # 3. Mark current drone position as free
        self.add_points(pose.position[0], pose.position[1], FREE_ZONE_VALUE)

        # Clip values
        self.grid = np.clip(self.grid, THRESHOLD_MIN, THRESHOLD_MAX)

    def display(self, pose: Pose, name="Mapping"):
        """Display the grid in a CV2 window."""
        img = self.grid.T
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)

        # Draw Drone Position
        pt_x, pt_y = self._conv_world_to_grid(pose.position[0], pose.position[1])
        if 0 <= pt_x < self.x_max_grid and 0 <= pt_y < self.y_max_grid:
            end_x = pt_x + 15 * np.cos(pose.orientation)
            end_y = pt_y - 15 * np.sin(pose.orientation)
            cv2.arrowedLine(img_color, (int(pt_x), int(pt_y)), (int(end_x), int(end_y)), (0, 0, 255), 2)
            cv2.circle(img_color, (int(pt_x), int(pt_y)), 4, (255, 255, 255), -1)

        display_img = cv2.resize(img_color, (600, 450), interpolation=cv2.INTER_NEAREST)
        cv2.imshow(name, display_img)
        cv2.waitKey(1)


class MyDroneEval(DroneAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.grid = OccupancyGrid(size_area_world=self.size_area,
                                  resolution=10.0,
                                  lidar=self.lidar())
        
        self.pose = Pose()
        self.use_odometry = False 

        # Navigation constants
        self.CRUISING_SPEED = 0.5     
        self.ROTATION_SPEED = 0.5     
        self.SAFETY_DISTANCE = 80
        self.STOP_DISTANCE = 40 

        self.angleStopTurning = 0
        self.isTurning = False

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

    def get_front_distance(self) -> float:
        """Returns distance to nearest obstacle in the front cone."""
        lidar_values = self.lidar_values()
        if lidar_values is None:
            return float('inf')
        
        # Look at a sector in front (-45 to +45 degrees approx)
        num_rays = len(lidar_values)
        center_idx = num_rays // 2
        fov_range = num_rays // 4  
        start_idx = max(0, center_idx - fov_range // 2)
        end_idx = min(num_rays, center_idx + fov_range // 2)
        
        front_sector = lidar_values[start_idx:end_idx]
        
        if len(front_sector) > 0:
            return min(front_sector)
        return float('inf')

    def get_turn_direction(self) -> int:
        """
        Decides whether to turn Left (+1) or Right (-1) using explicit ray angles.
        This is robust against sensor array ordering issues.
        """
        lidar_values = self.lidar_values()
        lidar_angles = self.lidar().ray_angles
        
        if lidar_values is None: return 1
        
        # Use Angle Limits to define Left vs Right sectors
        # Right Sector: Angles between -60 and 0 degrees
        # Left Sector: Angles between 0 and +60 degrees
        limit_rad = math.radians(60)
        
        # Create boolean masks
        # Note: We ignore 'far' readings to focus on nearby obstacles if needed,
        # but here we want to find open space, so we average all rays in sector.
        
        right_mask = (lidar_angles > -limit_rad) & (lidar_angles < 0)
        left_mask = (lidar_angles > 0) & (lidar_angles < limit_rad)
        
        right_values = lidar_values[right_mask]
        left_values = lidar_values[left_mask]
        
        avg_right = np.mean(right_values) if len(right_values) > 0 else 0
        avg_left = np.mean(left_values) if len(left_values) > 0 else 0
        
        # Logic: Turn towards the side with MORE space
        if avg_left > avg_right:
            return 1 # Turn Left (+)
        else:
            return -1 # Turn Right (-)

    def control(self) -> CommandsDict:
        # 1. Update Map
        self.update_pose()
        self.grid.update_grid(self.pose)
        
        if self.identifier == 0:
            title = "Mapping (GPS)" if not self.use_odometry else "Mapping (NO GPS - ODOMETRY)"
            self.grid.display(self.pose, name=title)

        # 2. Navigation Logic
        command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}
        
        dist_front = self.get_front_distance()

        # Check velocity to handle inertia
        velocity_vector = self.measured_velocity()
        current_speed = 0.0
        if velocity_vector is not None:
            drone_angle = self.pose.orientation
            current_speed = (velocity_vector[0] * math.cos(drone_angle) + 
                             velocity_vector[1] * math.sin(drone_angle))

        # State Machine
        if self.isTurning:
            # BRAKING LOGIC: Stop forward slide before turning
            if current_speed > 0.05: 
                command["forward"] = -0.5 # Reverse thrusters
                command["rotation"] = 0.0 
            else:
                # Perform the 10 degree turn
                command["forward"] = 0.0
                command["rotation"] = self.ROTATION_SPEED 
                
                diff_angle = normalize_angle(self.angleStopTurning - self.pose.orientation)
                
                # Apply rotation in the correct direction
                command["rotation"] = np.sign(diff_angle) * self.ROTATION_SPEED

                # Stop turning when we reach the 10 degree increment
                if abs(diff_angle) < 0.1:
                    self.isTurning = False

        elif dist_front < self.STOP_DISTANCE:
            # OBSTACLE TOO CLOSE -> DECIDE DIRECTION AND TURN 10 DEGREES
            self.isTurning = True
            
            # 1. Determine direction (+1 or -1)
            direction = self.get_turn_direction()
            
            # 2. Set target: Current angle + 10 degrees (in radians)
            turn_increment = math.radians(20)
            self.angleStopTurning = normalize_angle(self.pose.orientation + (direction * turn_increment))
            
            command["forward"] = 0.0 # Cut throttle immediately

        else:
            # MOVING FORWARD
            if dist_front > self.SAFETY_DISTANCE:
                command["forward"] = self.CRUISING_SPEED
            else:
                factor = (dist_front - self.STOP_DISTANCE) / (self.SAFETY_DISTANCE - self.STOP_DISTANCE)
                factor = max(0.1, min(1.0, factor))
                command["forward"] = self.CRUISING_SPEED * factor

        return command