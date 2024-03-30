#!/usr/bin/env python3
import rospy #type:ignore
from geometry_msgs.msg import PoseStamped #type:ignore
from nav_msgs.msg import Path #type:ignore
import numpy as np #type:ignore
from scipy.interpolate import splprep, splev #type:ignore
import matplotlib.pyplot as plt #type:ignore
from math import sqrt #type:ignore

class PathSmoothingNode:
    def __init__(self):
        rospy.init_node('path_smoothing_node')

        # Abonniere den globalen Pfad und veröffentliche den optimierten und geglätteten Pfad
        self.global_path_subscriber = rospy.Subscriber('/move_base_flex/GlobalPlanner/plan', Path, self.global_path_callback)
        self.direct_path_publisher = rospy.Publisher('/direct_path', Path, queue_size=10)
        self.optimized_path_publisher = rospy.Publisher('/optimized_path', Path, queue_size=10)

        self.douglas_peucker = DouglasPeucker(epsilon=0.5)  # Setzen Sie hier das gewünschte Epsilon
        self.curvature_analyzer = CurvatureAnalysis()

    def global_path_callback(self, path):
        # Glätte den globalen Pfad mithilfe des Douglas-Peucker-Algorithmus
        direct_path_poses = self.douglas_peucker.simplify(path.poses)
        direct_path_msg = Path(header=path.header, poses=direct_path_poses)
        self.direct_path_publisher.publish(direct_path_msg)

        # Führe die Krümmungsanalyse und -optimierung durch
        optimized_path_poses = self.optimize_path(direct_path_poses)

        # Veröffentliche die optimierten Pfade
        optimized_path_msg = Path(header=path.header, poses=optimized_path_poses)
        self.optimized_path_publisher.publish(optimized_path_msg)

        # Plotte die Pfade
        self.plot_paths(path, direct_path_poses, optimized_path_poses)

    def optimize_path(self, path_poses):
        optimized_poses = [path_poses[0]]  # Keep the start point unchanged
        i = 0
        while i < len(path_poses) - 2:
            angle = self.curvature_analyzer.calculate_angle(path_poses[i], path_poses[i+1], path_poses[i+2])
            if angle > self.curvature_analyzer.threshold_angle:
                spline_points = self.curvature_analyzer.spline_smoothing([path_poses[i-1], path_poses[i], path_poses[i+1], path_poses[i+2], path_poses[i+3]])
                optimized_poses.extend(spline_points[1:-1])  # Add the smoothed points (except the first and last points)
                i += 2  # Skip the next two points as they are already covered
            else:
                optimized_poses.append(path_poses[i+1])  # Add the next point
                i += 1
        optimized_poses.append(path_poses[-1])  # Keep the end point unchanged
        return optimized_poses

    def plot_paths(self, original_path, direct_path, optimized_path):
        original_x = [pose.pose.position.x for pose in original_path.poses]
        original_y = [pose.pose.position.y for pose in original_path.poses]

        direct_x = [pose.pose.position.x for pose in direct_path]
        direct_y = [pose.pose.position.y for pose in direct_path]

        optimized_x = [pose.pose.position.x for pose in optimized_path]
        optimized_y = [pose.pose.position.y for pose in optimized_path]

        plt.figure()
        plt.plot(original_x, original_y, label='Original Path', color='blue')
        plt.plot(direct_x, direct_y, label='Direct Path (Douglas-Peucker)', color='green')
        plt.plot(optimized_x, optimized_y, label='Optimized Path', color='red')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Path Visualization')
        plt.legend()
        plt.grid(True)
        plt.show()


class DouglasPeucker:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def distance(self, p1, p2, p):
        x1, y1 = p1.pose.position.x, p1.pose.position.y
        x2, y2 = p2.pose.position.x, p2.pose.position.y
        x, y = p.pose.position.x, p.pose.position.y
        numer = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
        denom = sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        return numer / denom

    def simplify(self, point_list):
        if len(point_list) < 3:
            return point_list
        dmax = 0
        index = 0
        end_index = len(point_list) - 1
        start_point = point_list[0]
        end_point = point_list[end_index]
        for i in range(1, end_index):
            d = self.distance(start_point, end_point, point_list[i])
            if d > dmax:
                index = i
                dmax = d
        if dmax > self.epsilon:
            left_segment = self.simplify(point_list[:index+1])
            right_segment = self.simplify(point_list[index:])
            return left_segment[:-1] + right_segment
        else:
            return [start_point, end_point]

class CurvatureAnalysis:
    def __init__(self):
        self.threshold_angle = 10  # Schwellenwert für den Krümmungswinkel in Grad

    def calculate_angle(self, p1, p2, p3):
        # Berechnung des Winkels zwischen den Vektoren p1p2 und p1p3
        vector1 = np.array([p2.pose.position.x - p1.pose.position.x, p2.pose.position.y - p1.pose.position.y])
        vector2 = np.array([p3.pose.position.x - p1.pose.position.x, p3.pose.position.y - p1.pose.position.y])
        dot_product = np.dot(vector1, vector2)
        norms_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        angle_rad = np.arccos(dot_product / norms_product)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    def spline_smoothing(self, points):
        # Extrahieren Sie x- und y-Koordinaten aus den Punkten
        x = np.array([pose.pose.position.x for pose in points])
        y = np.array([pose.pose.position.y for pose in points])

        # Glättung mit B-Splines
        tck, _ = splprep([x, y], k=3)
        u = np.linspace(0, 1, num=50, endpoint=True)
        spline_points = splev(u, tck)
        spline_pose_list = [PoseStamped() for _ in range(len(spline_points[0]))]
        for i in range(len(spline_pose_list)):
            spline_pose_list[i].pose.position.x = spline_points[0][i]
            spline_pose_list[i].pose.position.y = spline_points[1][i]
        return spline_pose_list

if __name__ == '__main__':
    try:
        path_smoothing_node = PathSmoothingNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass