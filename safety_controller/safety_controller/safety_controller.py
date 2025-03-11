#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import math
import numpy as np


class SafetyController(Node):
    def __init__(self):
        super().__init__('safety_controller')

        # Declare params
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('drive_topic_in', '/drive_raw')
        self.declare_parameter('drive_topic_out', '/drive')
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('look_ahead_dist', 0.75) #TUNE: 0.75
        self.declare_parameter('car_width', 0.32)
        self.declare_parameter('max_points_in_envelope', 2)

        # Get params
        self.scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.drive_topic_in = self.get_parameter('drive_topic_in').get_parameter_value().string_value
        self.drive_topic_out = self.get_parameter('drive_topic_out').get_parameter_value().string_value
        self.wheelbase = self.get_parameter('wheelbase').get_parameter_value().double_value
        self.look_ahead_dist = self.get_parameter('look_ahead_dist').get_parameter_value().double_value
        self.car_width = self.get_parameter('car_width').get_parameter_value().double_value
        self.max_points = self.get_parameter('max_points_in_envelope').get_parameter_value().integer_value

        # ROS subscribers and publishers
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.lidar_callback, 10)
        self.drive_sub = self.create_subscription(AckermannDriveStamped, self.drive_topic_in, self.desired_drive_callback, 10)
        self.safety_drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic_out, 10)

        #Cache last drive command received
        self.last_desired_drive = AckermannDriveStamped()


    def desired_drive_callback(self, msg: AckermannDriveStamped):
        """
        Cache the latest drive command from wall follower
        """
        self.last_desired_drive = msg

    def lidar_callback(self, scan_msg: LaserScan):
        """
        1. Get steering angle and velocity from last drive message
        2. Build an envelope shape in the robot frame
        3. Convert each LIDAR point to (x,y) and check if inside that envelope
        4. If more than max points points are inside then stop. Otherwise pass the drive
        """

        # Get speed and steering angle
        speed = self.last_desired_drive.drive.speed
        steering_angle = self.last_desired_drive.drive.steering_angle

        # No collision checking if speed ~0
        if abs(speed) < 1e-2:
            self.publish_safe_drive(self.last_desired_drive)
            return

        # LIDAR data processing
        angle_min = scan_msg.angle_min
        angle_inc = scan_msg.angle_increment
        ranges = np.array(scan_msg.ranges)
        angles = angle_min + angle_inc * np.arange(len(ranges))

        # Convert to cartesian in robot frame (x-forward, y-left)
        xs, ys = [], []
        for r, th in zip(ranges, angles):
            if math.isfinite(r) and r > 0.0:
                x = r * math.cos(th)
                y = r * math.sin(th)
                xs.append(x)
                ys.append(y)

        # Initialize counter for LIDAR points in collision envelope
        points_in_envelope = 0
        half_width = self.car_width / 2.0

        # Point checker for straight driving
        if abs(steering_angle) < 1e-3:
            for x, y in zip(xs, ys):
                if x < 0.0:
                    continue
                if x > self.look_ahead_dist:
                    continue
                if abs(y) > half_width:
                    continue
                points_in_envelope += 1
                if points_in_envelope > self.max_points:
                    break

        # Point checker for curved driving
        else:
            R = self.wheelbase / math.tan(steering_angle)
            arc_sweep = self.look_ahead_dist / abs(R)
            cx = 0.0
            cy = R
            sign_R = 1.0 if R >= 0 else -1.0
            for x, y in zip(xs, ys):
                dist_c = math.sqrt((x - cx)**2 + (y - cy)**2)
                if not (abs(R) - half_width <= dist_c <= abs(R) + half_width):
                    continue
                if x < -0.1:
                    continue
                theta_p = math.atan2(y - cy, x - cx)
                angle_car = math.atan2(-R, 0.0)
                delta_angle = self.normalize_angle(theta_p - angle_car)
                if sign_R > 0:
                    if delta_angle < 0.0 or delta_angle > arc_sweep:
                        continue
                else:
                    if delta_angle > 0.0 or delta_angle < -arc_sweep:
                        continue
                points_in_envelope += 1
                if points_in_envelope > self.max_points:
                    break

        # If number of points in the envelope exceeds threshold issue stop command
        if points_in_envelope > self.max_points:
            self.get_logger().warn(f"COLLISION DETECTED: {points_in_envelope} points => STOP.")
            safe_cmd = AckermannDriveStamped()
            safe_cmd.drive.speed = 0.0
            safe_cmd.drive.steering_angle = 0.0
            self.publish_safe_drive(safe_cmd)

        # Otherwise pass through last desired drive command
        else:
            self.publish_safe_drive(self.last_desired_drive)


    def publish_safe_drive(self, drive_msg: AckermannDriveStamped):
        """
        Publishes provided drive message to safety drive topic
        """
        self.safety_drive_pub.publish(drive_msg)


    def normalize_angle(self, angle):
        """
        Normalize an angle to range [-pi, pi]
        """
        res = math.atan2(math.sin(angle), math.cos(angle))
        return res


def main(args=None):
    rclpy.init(args=args)
    node = SafetyController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
