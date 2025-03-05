#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from rcl_interfaces.msg import SetParametersResult

from wall_follower.visualization_tools import VisualizationTools


class WallFollower(Node):

    def __init__(self):
        super().__init__("wall_follower")
        # Declare parameters to make them available for use
        # DO NOT MODIFY THIS!
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("drive_topic", "/vesc/high_level/ackermann_cmd")
        self.declare_parameter("side", -1)
        self.declare_parameter("velocity", 0.5)
        self.declare_parameter("desired_distance", 0.5)

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

	    # TODO: Initialize your publishers and subscribers here
        self.publisher_ = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 10)
        self.line_publisher = self.create_publisher(Marker, 'marker', 10)
        self.subscription = self.create_subscription(LaserScan, self.SCAN_TOPIC, self.listener_callback, 10)
        self.subscription

        self.prev_error = 0
        self.t_before = 0

    # TODO: Write your callback functions here
    def listener_callback(self, laser_scan):
        ranges = np.array(laser_scan.ranges) # array of distances from lidar sensor to nearest obstacle
        angle_increment = laser_scan.angle_increment

        ack_msg = AckermannDriveStamped()
        ack_msg.header.stamp = self.get_clock().now().to_msg()

        # Slicing ranges, get x y for wall
        # mid_index = int(np.floor(len(ranges)/2))
        mid_index = int((2/3*np.pi + np.pi/6) / angle_increment)

        offset = self.DESIRED_DISTANCE

        if self.SIDE  == 1: # left, get top left quarter circle
            # mid_index = int((2/3*np.pi - np.pi/6) / angle_increment)
            right_index = int((2/3*np.pi + np.pi/2) / angle_increment)
            start_angle = np.pi/2
            new_range = ranges[mid_index:right_index]

            index_array = np.arange(0, len(new_range))
            angle_array = angle_increment*index_array
            wall_y = new_range*np.sin(angle_array)
            wall_x = new_range*np.cos(angle_array)

            offset*=-1
        else: # right, get top right quarter circle
            # mid_index = int((2/3*np.pi + np.pi/6) / angle_increment)
            left_index = int((2/3*np.pi - np.pi/2) / angle_increment)
            start_angle = -np.pi/2
            new_range = ranges[mid_index:left_index:-1]

            index_array = np.arange(0, len(new_range))
            angle_array = angle_increment*index_array
            wall_y = -new_range*np.sin(angle_array)
            wall_x = new_range*np.cos(angle_array)


        # TODO: TUNE THE STARTING ANGLE INSTEAD OF 0 SO THAT IT CAN MAKE CLOSER TURNS

        # polyfit the wall
        m, b = np.polyfit(wall_x, wall_y, 1)
        lookahead_x = np.arange(0, 2.0, 5/20, dtype = float) # 2, 5/20
        desired_y = m*lookahead_x + b + offset

        # Plot path offset from wall
        VisualizationTools.plot_line(lookahead_x, desired_y, self.line_publisher, frame = "/laser")

        # PID controller

        t_after = self.get_clock().now().nanoseconds /(1e9)
        dt = t_after - self.t_before
        self.t_before = self.get_clock().now().nanoseconds / (1e9)

        error = np.mean(desired_y)

        self.get_logger().info("wall_y: %s, desired_y: %s" %(wall_y, desired_y))

        integral = error*dt
        deriv = (self.prev_error - error)/dt
        self.prev_error = error

        kp = 5
        ki = 0
        kd = 0

        steering_angle = kp*error + ki*integral - kd*deriv

        ack_msg.drive.steering_angle = steering_angle
        ack_msg.drive.speed = np.abs(self.VELOCITY)

        self.publisher_.publish(ack_msg)
        self.get_logger().info('I hear then publish \n steering angle: "%s" \n speed: "%s"\n'
                               % (ack_msg.drive.steering_angle, ack_msg.drive.speed))

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


def main():
    rclpy.init()
    wall_follower = WallFollower()
    rclpy.spin(wall_follower)
    wall_follower.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
