#!/usr/bin/env python3
import rospy #type:ignore
from geometry_msgs.msg import PoseStamped #type:ignore
from nav_msgs.msg import Path #type:ignore
import numpy as np #type:ignore
from scipy.interpolate import splprep, splev #type:ignore
import matplotlib.pyplot as plt #type:ignore

class PathSmoothing:
    def __init__(self):
        rospy.init_node('path_smoothing_node', anonymous=True)
        self.global_path_subscriber = rospy.Subscriber('/move_base_flex/GlobalPlanner/plan', Path, self.global_path_callback)
        self.optimize_path_publisher = rospy.Publisher('/optimize_path', Path, queue_size=10)
        
    def global_path_callback(self, msg):
        # Extracting path points
        path_points = np.array([(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses])
        print("Original path points:", len(path_points))

        # Smoothing path using splprep
        tck, _ = splprep(path_points.T, s=0.5)
        smoothed_path_points = splev(np.linspace(0, 1, 1000), tck)
        print("Smoothed path points:", len(smoothed_path_points[0]))

        # Publish the optimized path
        optimized_path_msg = Path()
        optimized_path_msg.header = msg.header
        optimized_path_msg.poses = [self.create_pose(p[0], p[1]) for p in np.array(smoothed_path_points).T]
        self.optimize_path_publisher.publish(optimized_path_msg)

        # Plot original and optimized paths
        self.plot_paths(path_points, np.array(smoothed_path_points).T)

    def create_pose(self, x, y):
        pose = PoseStamped()
        pose.pose.position.x = x
        pose.pose.position.y = y
        return pose
    
    def plot_paths(self, original_path, smoothed_path):
        plt.figure()
        plt.plot(original_path[:, 0], original_path[:, 1], 'r-', label='Original Path')
        plt.plot(smoothed_path[:, 0], smoothed_path[:, 1], 'b-', label='Smoothed Path')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Original vs Smoothed Path')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    path_smoothing = PathSmoothing()
    rospy.spin()

