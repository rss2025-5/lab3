#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from rcl_interfaces.msg import SetParametersResult

from wall_follower.visualization_tools import VisualizationTools


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


class WallFollower(Node):

    def __init__(self):
        super().__init__("wall_follower")
        # Declare parameters to make them available for use
        # DO NOT MODIFY THIS! 
        self.declare_parameter("scan_topic", "default")
        self.declare_parameter("drive_topic", "default")
        self.declare_parameter("side", "default")
        self.declare_parameter("velocity", "default")
        self.declare_parameter("desired_distance", "default")

        # Fetch constants from the ROS parameter server
        # DO NOT MODIFY THIS! This is necessary for the tests to be able to test varying parameters!
        self.SCAN_TOPIC = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.DRIVE_TOPIC = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.SIDE = self.get_parameter('side').get_parameter_value().integer_value
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
        self.DESIRED_DISTANCE = self.get_parameter('desired_distance').get_parameter_value().double_value
		
        # This activates the parameters_callback function so that the tests are able
        # to change the parameters during testing.
        # DO NOT MODIFY THIS! 
        self.add_on_set_parameters_callback(self.parameters_callback)
  

        self.scan_sub = self.create_subscription(LaserScan, self.SCAN_TOPIC, self.scan_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 10)

        self.line_pub = self.create_publisher(Marker, "/wall", 1)
        self.prev_error = 0
        self.integral_error = 0
        self.total_loss = 0
        self.num_loss = 0

    def parameters_callback(self, params):
        """
        DO NOT MODIFY THIS CALLBACK FUNCTION!
        
        This is used by the test cases to modify the parameters during testing. 
        It's called whenever a parameter is set via 'ros2 param set'.
        """
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
        return SetParametersResult(successful=True)

    def scan_callback(self, msg):
        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        distances = np.array(msg.ranges)
        assert len(distances) == len(angles), f"len(distances) = {len(distances)}, len(angles) = {len(angles)}"
        scan_polar_vectors = np.vstack((distances, angles))

        side_cut = 40
        side_min_angle = np.radians(90 - side_cut/2)
        side_max_angle = np.radians(90 + side_cut/2)

        def side_wall_filter(side):
            if side == 1:
                side_wall = scan_polar_vectors[:, scan_polar_vectors[1,:] >= 0]
            else:
                side_wall = scan_polar_vectors[:, scan_polar_vectors[1,:] <= 0]
            side_wall = side_wall[ 
                :,  (side_min_angle <= np.abs(side_wall[1,:])) & 
                    (np.abs(side_wall[1,:]) <= side_max_angle)
            ]

            max_scan_dist = 3 * max(self.VELOCITY, self.DESIRED_DISTANCE)
            side_wall = side_wall[:, side_wall[0,:] <= max_scan_dist]

            return side_wall

        left_wall = side_wall_filter(1)
        right_wall = side_wall_filter(-1)
        tracked_wall = left_wall if self.SIDE == 1 else right_wall

        if tracked_wall.size == 0:
            self.drive(self.VELOCITY, 0.0)
            return

        m, b, dist = interpolate_wall_points(
            tracked_wall[0], tracked_wall[1]
        )

        error = dist - self.DESIRED_DISTANCE
        self.total_loss += abs(error)
        self.num_loss += 1 

        self.get_logger().info("Loss is %f" % self.score)
        P = 17 * error
        hz = 1/msg.time_increment if msg.time_increment > 0 else 20
        D = 2 * (self.prev_error - error) * hz
        I = 0.2 * self.integral_error

        # Calculates the steering angle.
        steering_angle = np.clip(
            self.SIDE * (P + I - D), np.radians(-90), np.radians(90)
        )
        f_cut = 15
        forward_cutoff = np.clip(
            msg.angle_increment * 50,
            -np.radians(f_cut/2), np.radians(f_cut/2)
        )
        forward_wall = scan_polar_vectors[
            :,  (scan_polar_vectors[1,:] >= -forward_cutoff) & 
                (scan_polar_vectors[1,:] <= forward_cutoff)
        ]
        turn_rad = 0.7
        e_stop_dist = (self.DESIRED_DISTANCE + turn_rad) + self.VELOCITY/hz
        f_wall_pts = len(forward_wall[0,:])
        forward_wall = forward_wall[:, forward_wall[0,:] <= 2 * e_stop_dist]
        if len(forward_wall[0]) < 0.6 * f_wall_pts:
            m = 0
            b = 0
            dist = np.inf
        else:
            m, b, dist = interpolate_wall_points(
                forward_wall[0], forward_wall[1]
            )
        km = 150
        if dist <= e_stop_dist and np.abs(m) > 0.005:
            steering_angle = np.clip(
                -self.SIDE * km * self.VELOCITY/dist * np.abs(m),
                -np.radians(90), np.radians(90)
            )
            error = (dist - self.DESIRED_DISTANCE) / hz

        self.prev_error = error
        self.integral_error += error / hz
        self.integral_error %= 1 * self.DESIRED_DISTANCE

        self.drive(self.VELOCITY, steering_angle)



    def draw_wall(self, x, y):
 
        VisualizationTools.plot_line(x, y, self.line_pub, frame="/laser")


    def drive(self, speed, angle):

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()

        drive_msg.drive.speed = speed
        drive_msg.drive.steering_angle = angle

        self.drive_pub.publish(drive_msg)


def main():
    rclpy.init()
    wall_follower = WallFollower()
    rclpy.spin(wall_follower)
    wall_follower.destroy_node()
    rclpy.shutdown()


if __name__ == '_main_':
    main()
    