#!/usr/bin/env python3
import numpy as np
import rclpy
import math
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from rcl_interfaces.msg import SetParametersResult
from visualization_msgs.msg import Marker
# from wall_follower.visualization_tools import VisualizationTools


class WallFollower(Node):
    def __init__(self):
        super().__init__("wall_follower")

        # Declare params
        self.declare_parameter("scan_topic", "default")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("side", 1)
        self.declare_parameter("velocity", 1.0)
        self.declare_parameter("desired_distance", 1.0)
        self.declare_parameter("lookahead_distance", 1.5) #TUNE: 1.5
        self.declare_parameter("wheelbase", 0.33)
        self.declare_parameter("forward_half_deg", 12.5) #TUNE: 12.5
        self.declare_parameter("corner_dist", 1.5) #TUNE: 1.5
        self.declare_parameter("corner_steering_offset", -0.75) #TUNE: -0.75

        # Get params
        self.SCAN_TOPIC = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.DRIVE_TOPIC = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.SIDE = self.get_parameter('side').get_parameter_value().integer_value
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
        self.DESIRED_DISTANCE = self.get_parameter('desired_distance').get_parameter_value().double_value
        self.LOOKAHEAD_DISTANCE = self.get_parameter('lookahead_distance').get_parameter_value().double_value
        self.WHEELBASE = self.get_parameter('wheelbase').get_parameter_value().double_value
        self.FORWARD_HALF_DEG = self.get_parameter('forward_half_deg').get_parameter_value().double_value
        self.CORNER_DIST = self.get_parameter('corner_dist').get_parameter_value().double_value
        self.CORNER_STEERING_OFFSET = self.get_parameter('corner_steering_offset').get_parameter_value().double_value

        # Dynamic updates
        self.add_on_set_parameters_callback(self.parameters_callback)

        # Default topics
        if self.SCAN_TOPIC == "default":
            self.SCAN_TOPIC = "/scan"
        if self.DRIVE_TOPIC == "default":
            self.DRIVE_TOPIC = "/drive"

        # ROS subscribers and publishers
        self.scan_sub = self.create_subscription(LaserScan, self.SCAN_TOPIC, self.lidar_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 10)


    def lidar_callback(self, scan_msg: LaserScan):
        """
        1. Perform line-fitting on a slice of LIDAR data to find wall
        2. Compute a lookahead point on an offset line
        3. Check if there's a forward internal corner and add extra steering offset if so
        4. Publish the drive command
        """

        # LIDAR data processing
        angle_min = scan_msg.angle_min
        angle_inc = scan_msg.angle_increment
        ranges = np.array(scan_msg.ranges)
        n = len(ranges)

        # Extract slice of LIDAR data around side wall
        center_angle_deg = 70.0 * float(self.SIDE) #TUNE: 70
        half_window_deg  = 20.0 #TUNE: 20
        def angle_to_index(a):
            return int(round((a - angle_min) / angle_inc))
        start_angle = math.radians(center_angle_deg - half_window_deg)
        end_angle = math.radians(center_angle_deg + half_window_deg)
        start_i = max(0, min(n - 1, angle_to_index(start_angle)))
        end_i = max(0, min(n - 1, angle_to_index(end_angle)))
        if start_i > end_i:
            start_i, end_i = end_i, start_i
        angles = angle_min + angle_inc * np.arange(start_i, end_i + 1)
        selected_ranges = ranges[start_i:end_i + 1]

        # Convert to cartesian in robot frame (x-forward, y-left)
        xs, ys = [], []
        for r, th in zip(selected_ranges, angles):
            if not math.isfinite(r) or r <= 0.0:
                continue
            x = r * math.cos(th)
            y = r * math.sin(th)
            xs.append(x)
            ys.append(y)
        xs = np.array(xs)
        ys = np.array(ys)
        if len(xs) < 2:
            # If not enough valid points drive straight
            self.publish_drive(0.0)
            return

        # Fit a line (y = m*x + c) to points with simple linear regression
        x_mean = np.mean(xs)
        y_mean = np.mean(ys)
        num = np.sum((xs - x_mean) * (ys - y_mean))
        den = np.sum((xs - x_mean)**2)
        if abs(den) < 1e-7:
            # If degenerate case drive straight
            self.publish_drive(0.0)
            return
        m = num / den
        c = y_mean - m * x_mean

        # Offset line to maintain desired distance
        offset_sign = -1.0 if (self.SIDE > 0) else 1.0
        dist_term = self.DESIRED_DISTANCE * math.sqrt(1 + m*m)
        c_offset = c + offset_sign * dist_term

        # Pick a "lookahead point" on that offset line
        x_L = self.LOOKAHEAD_DISTANCE
        y_L = m * x_L + c_offset

        # Pure pursuit steering
        alpha = math.atan2(y_L, x_L)
        Ld = math.sqrt(x_L*x_L + y_L*y_L)
        if Ld < 1e-7:
            self.publish_drive(0.0)
            return
        steering_angle = math.atan2(2.0 * self.WHEELBASE * math.sin(alpha), Ld)

        # # Check if there's an internal corner and add steering offset
        corner_offset = self.check_forward_obstacle(scan_msg)
        steering_angle += corner_offset

        # Publish  drive command
        self.publish_drive(steering_angle)


    def check_forward_obstacle(self, scan_msg):
        """
        Check a small forward window.
        If the min distance is < self.corner_dist, return a steering offset.
        Return 0.0 if no corner is detected.
        """

        # LIDAR data processing
        angle_min = scan_msg.angle_min
        angle_inc = scan_msg.angle_increment
        ranges = np.array(scan_msg.ranges)
        n = len(ranges)

        # Extract slice of LIDAR ahead of car
        half_rad = math.radians(self.FORWARD_HALF_DEG)
        def angle_to_index(a):
            return int(round((a - angle_min) / angle_inc))
        start_a = -half_rad
        end_a  =  half_rad
        start_i = max(0, min(n - 1, angle_to_index(start_a)))
        end_i = max(0, min(n - 1, angle_to_index(end_a)))
        if start_i > end_i:
            start_i, end_i = end_i, start_i

        # Filter to get valid points
        front_slice = ranges[start_i:end_i + 1]
        valid = [r for r in front_slice if math.isfinite(r) and r > 0.0]
        if not valid:
            return 0.0

        # Check for min distance and apply extra steering if needed
        min_front_dist = min(valid)
        if min_front_dist < self.CORNER_DIST:
            return self.CORNER_STEERING_OFFSET * float(self.SIDE)
        else:
            return 0.0

    def publish_drive(self, angle):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = float(self.VELOCITY)
        drive_msg.drive.steering_angle = float(angle)
        self.drive_pub.publish(drive_msg)

    def parameters_callback(self, params):
        for param in params:
            if param.name == 'side':
                self.SIDE = param.value
                self.get_logger().info(f"Updated side to {self.SIDE}")
            elif param.name == 'velocity':
                self.VELOCITY = param.value
                self.get_logger().info(f"Updated velocity to {self.VELOCITY}")
            elif param.name == 'desired_distance':
                self.DESIRED_DISTANCE = param.value
                self.get_logger().info(f"Updated desired_distance to {self.DESIRED_DISTANCE}")
            elif param.name == 'lookahead_distance':
                self.LOOKAHEAD_DISTANCE = param.value
                self.get_logger().info(f"Updated lookahead_distance to {self.LOOKAHEAD_DISTANCE}")
            elif param.name == 'wheelbase':
                self.WHEELBASE = param.value
                self.get_logger().info(f"Updated wheelbase to {self.WHEELBASE}")
            elif param.name == 'forward_half_deg':
                self.FORWARD_HALF_DEG = param.value
                self.get_logger().info(f"Updated forward_half_deg to {self.FORWARD_HALF_DEG}")
            elif param.name == 'corner_dist':
                self.CORNER_DIST = param.value
                self.get_logger().info(f"Updated corner_dist to {self.CORNER_DIST}")
            elif param.name == 'corner_steering_offset':
                self.CORNER_STEERING_OFFSET = param.value
                self.get_logger().info(f"Updated corner_steering_offset to {self.CORNER_STEERING_OFFSET}")
        return SetParametersResult(successful=True)


def main():
    rclpy.init()
    wall_follower = WallFollower()
    rclpy.spin(wall_follower)
    wall_follower.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
