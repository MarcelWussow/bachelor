#!/usr/bin/env python3
import rospy #type:ignore
from nav_msgs.msg import Path #type:ignore
import numpy as np #type:ignore
from scipy.interpolate import splprep, splev #type:ignore
from geometry_msgs.msg import PoseStamped #type:ignore
import math #type:ignore

#erstmal irrelevant

class GlobalPathOptimizationNode:
    """
    A class for optimizing global paths for a two-wheeled robot (e.g., MiR200) considering non-holonomic constraints
    and geometric and kinematic properties.
    """

    def __init__(self):
        """
        Initializes the GlobalPathOptimizationNode.
        """
        rospy.init_node('global_path_optimization_node')

        # ROS Parameters for robot properties
        self.robot_radius = rospy.get_param('~robot_radius', 0.5)  # Default: 0.5 meters
        self.max_steering_angle = rospy.get_param('~max_steering_angle', math.pi / 4)  # Default: 45 degrees
      
        # Subscribers and Publishers
        self.global_path_subscriber = rospy.Subscriber('/global_planner/path', Path, self.global_path_callback)
        self.optimized_path_publisher = rospy.Publisher('/optimized_path', Path, queue_size=10)

    def global_path_callback(self, path_msg):
        """
        Callback function for processing the global path.
        :param path_msg: The received global path message.
        """

        # Extract points from the path message
        path = np.array([(pose.pose.position.x, pose.pose.position.y) for pose in path_msg.poses])

        # Find the index of the point with the smallest radius of curvature
        min_radius_index = self.calculate_min_radius_index(path)

        # Generate a new path by selecting the segment with the minimum curvature radius
        optimized_path = path[min_radius_index:]

        # Smooth the new path using a 3rd order Bezier curve
        tck, u = splprep(optimized_path.T, k=3, s=10)
        smoothed_path = splev(np.linspace(0, 1, len(optimized_path) * 2), tck)

        # Create a new path message for the optimized path
        optimized_path_msg = Path()
        optimized_path_msg.header = path_msg.header
        for x, y in zip(smoothed_path[0], smoothed_path[1]):
            pose = PoseStamped()
            pose.pose.position.x = x
            pose.pose.position.y = y
            optimized_path_msg.poses.append(pose)

        # Publish the optimized path
        self.optimized_path_publisher.publish(optimized_path_msg)

    def calculate_min_radius_index(self, path):
        """
        Calculates the index of the point on the path with the smallest curvature radius.
        :param path: Array of points representing the path.
        :return: Index of the point with the smallest curvature radius.
        """
        min_radius = float('inf')
        min_radius_index = 0
        for i in range(1, len(path) - 1):
            # Calculate curvature radius for each point, considering the maximum steering angle
            radius = self.robot_radius / math.sin(min(self.max_steering_angle, self.calculate_steering_angle(path[i-1], path[i], path[i+1])))
            if radius < min_radius:
                min_radius = radius
                min_radius_index = i
        return min_radius_index

    def calculate_steering_angle(self, p0, p1, p2):
        """
        Calculate the steering angle given three consecutive points.
        """
        angle1 = math.atan2(p1[1]-p0[1], p1[0]-p0[0])
        angle2 = math.atan2(p2[1]-p1[1], p2[0]-p1[0])
        return angle2 - angle1


if __name__ == '__main__':
    try:
        GlobalPathOptimizationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
