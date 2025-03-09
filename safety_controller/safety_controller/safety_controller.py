#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from rcl_interfaces.msg import SetParametersResult

from mse_msgs.msg import MSE


def polar_to_cartesian(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Convert polar coordinates to cartesian coordinates
    Args:
        r: radius
        theta: angle
    Returns:
        x, y
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def interpolate_wall_points(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Interpolate wall points
    Args:
        r: radius
        theta: angle
    Returns:
        x, y, dist
    """
    x, y = polar_to_cartesian(r, theta)
    # Runs linear regression to determine the slope of the wall.
    m, b = np.polyfit(x, y, 1)
    # Calculates the distance to the wall.
    distance = np.abs(b) / np.sqrt(1 + m**2)

    return m, b, distance

class SafetyController(Node):

    def __init__(self):
        super().__init__('safety_controller')

         # Declare parameters to make them available for use
        # DO NOT MODIFY THIS!
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("safety_topic", "/vesc/low_level/input/safety")
        self.declare_parameter("intercept_topic", "default")
        self.declare_parameter("stop_distance", 0.5)

        # Fetch constants from the ROS parameter server
        # DO NOT MODIFY THIS! This is necessary for the tests to be able to test varying parameters!
        self.SCAN_TOPIC = self.get_parameter('scan_topic').get_parameter_value().string_value

        self.get_logger().info(self.SCAN_TOPIC)

        self.SAFETY_TOPIC = "/vesc/low_level/input/safety"
        self.INTERCEPT_TOPIC = "/vesc/low_level/input/navigation"
        self.STOP_DISTANCE = 0.3

        #self.intercept_sub = self.create_subscription(AckermannDriveStamped, self.INTERCEPT_TOPIC, 10)
        self.safety_pub = self.create_publisher(AckermannDriveStamped, self.SAFETY_TOPIC, 10)
        self.scan_sub = self.create_subscription(LaserScan, self.SCAN_TOPIC, self.scan_callback, 10)
        self.intercept_sub = self.create_subscription(AckermannDriveStamped, self.INTERCEPT_TOPIC, self.stopped_callback, 10)
        self.mse_pub = self.create_publisher(MSE, "mse", 10)
        self.mse_sub = self.create_subscription(MSE, "mse", self.mse_error_retrieve, 10)


        self.mse_total_error = 0
        self.mse_num_errors = 0
        self.mse = 0

    def scan_callback(self, scan):
        angles = np.arange(scan.angle_min, scan.angle_max, scan.angle_increment)
        distances = np.array(scan.ranges)
        scan_polar_vectors = np.vstack((distances, angles))

        f_cut = 60
        forward_cutoff = np.clip(
            scan.angle_increment * 2,
            -np.radians(f_cut/2), np.radians(f_cut/2)
        )
        forward_wall = scan_polar_vectors[
            :,  (scan_polar_vectors[1,:] >= -forward_cutoff) &
                (scan_polar_vectors[1,:] <= forward_cutoff)
        ]
        f_wall_pts = len(forward_wall[0,:])
        forward_wall = forward_wall[:, forward_wall[0,:] <= 4*self.STOP_DISTANCE]
        if len(forward_wall[0]) < 0.6 * f_wall_pts or len(forward_wall[0]) == 0:
            m = 0
            b = 0
            dist = np.inf
        else:
            m, b, dist = interpolate_wall_points(
                    forward_wall[0], forward_wall[1]
                )
        if dist <= 2*self.STOP_DISTANCE:
             self.drive(0.0, 0.0)


    def drive(self, speed, angle):

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()

        drive_msg.drive.speed = speed
        drive_msg.drive.steering_angle = angle

        self.safety_pub.publish(drive_msg)

    def mse_error_retrieve(self, MSE_msg):
        self.mse_num_errors = MSE_msg.num_errors
        self.mse_total_error = MSE_msg.total_error

    def stopped_callback(self, ack_cmd):
        if ack_cmd.drive.speed == 0:
            mse = self.mse_total_error/self.mse_num_errors
            self.get_logger().info("MSE: %s" %mse)

def main(args=None):
    rclpy.init(args=args)
    safety_controller = SafetyController()
    rclpy.spin(safety_controller)
    safety_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
