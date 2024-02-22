#!/usr/bin/env python3
import rospy #type:ignore
from nav_msgs.msg import Path #type:ignore
import numpy as np #type:ignore
from scipy.interpolate import CubicBezier #type:ignore
from geometry_msgs.msg import PoseStamped #type:ignore

class GlobalPathOptimizationNode:
    def __init__(self):
        rospy.init_node('global_path_optimization_node')
        
        # Roboterparameter
        self.robot_radius = rospy.get_param('~robot_radius', 0.5)
        self.max_steering_angle = rospy.get_param('~max_steering_angle', np.pi / 4)
        self.smoothing_factor = rospy.get_param('~smoothing_factor', 10)

        # Abonnieren des globalen Pfads
        self.global_path_subscriber = rospy.Subscriber('/global_planner/path', Path, self.global_path_callback)
        # Veröffentlichen des optimierten Pfads
        self.optimized_path_publisher = rospy.Publisher('/optimized_path', Path, queue_size=10)

    def global_path_callback(self, path_msg):
        path = np.array([(pose.pose.position.x, pose.pose.position.y) for pose in path_msg.poses])
        
        # Identifiziere Kurvenabschnitte im Pfad
        curve_indices = self.identify_curve_sections(path)

        # Passe Kurven mit Dubin-Modell an
        adjusted_path = self.adjust_curves_with_dubins(path, curve_indices)

        # Glätte den Pfad basierend auf C1-Stetigkeit
        smoothed_path = self.ensure_c1_continuity(adjusted_path)

        # Veröffentliche den optimierten Pfad
        optimized_path_msg = Path()
        optimized_path_msg.header = path_msg.header

        for point in smoothed_path:
            pose = PoseStamped()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            optimized_path_msg.poses.append(pose)

        self.optimized_path_publisher.publish(optimized_path_msg)

    def identify_curve_sections(self, path):
        curve_indices = []
        for i in range(1, len(path) - 1):
            p0, p1, p2 = path[i-1:i+2]
            angle = self.calculate_curvature_angle(p0, p1, p2)
            if abs(angle) > 0.1:
                curve_indices.append(i)
        return curve_indices

    def calculate_curvature_angle(self, p0, p1, p2):
        v1 = np.array(p1) - np.array(p0)
        v2 = np.array(p2) - np.array(p1)
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        return angle

    def adjust_curves_with_dubins(self, path, curve_indices):
        adjusted_path = path.copy()
        for i in curve_indices:
            if i > 0 and i < len(path) - 1:
                p0, p1, p2 = path[i-1:i+2]
                adjusted_curve_point = self.calculate_dubins_curve_point(p0, p1, p2)
                adjusted_path[i] = adjusted_curve_point
        return adjusted_path

    def calculate_dubins_curve_point(self, p0, p1, p2):
        adjusted_point = ((p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2)

        v1 = np.array(p1) - np.array(p0)
        v2 = np.array(p2) - np.array(p1)
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        min_turning_radius = self.robot_radius / np.sin(min(angle, self.max_steering_angle))

        distance_p0_p2 = np.linalg.norm(np.array(p2) - np.array(p0))

        if distance_p0_p2 < min_turning_radius:
            scale_factor = min_turning_radius / distance_p0_p2
            adjusted_point = (p0[0] + scale_factor * v1[0], p0[1] + scale_factor * v1[1])

        return adjusted_point

    def ensure_c1_continuity(self, path):
        smoothed_path = []
        for i in range(len(path) - 1):
            smoothed_path.extend(self.bezier_smooth_segment(path[i], path[i + 1]))
        smoothed_path.append(path[-1])  # Add the last point

        # Überprüfen und Glätten, um C1-Stetigkeit sicherzustellen
        if not self.check_c1_continuity(smoothed_path):
            smoothed_path = self.bezier_smooth_segment(smoothed_path[0], smoothed_path[-1])

        return smoothed_path

    def bezier_smooth_segment(self, p0, p1):
        # Bezier interpolation für den Segment
        x = np.array([p0[0], p1[0]])
        y = np.array([p0[1], p1[1]])
        t = np.linspace(0, 1, self.smoothing_factor)
        curve = CubicBezier(x[0], y[0], x[-1], y[-1], 0, 0, 0, 0)
        smoothed_segment = curve(t)
        return smoothed_segment

    def check_c1_continuity(self, path):
        for i in range(1, len(path) - 1):
            p0, p1, p2 = path[i-1:i+2]
            v1 = np.array(p1) - np.array(p0)
            v2 = np.array(p2) - np.array(p1)
            if not np.allclose(v1, v2):
                return False
        return True

if __name__ == '__main__':
    try:
        GlobalPathOptimizationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
