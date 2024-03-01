#!/usr/bin/env python3
import rospy #type:ignore
from nav_msgs.msg import Path #type:ignore
import numpy as np #type:ignore
from scipy.interpolate import splprep, splev #type:ignore
from geometry_msgs.msg import PoseStamped #type:ignore
import matplotlib.pyplot as plt #type:ignore

class GlobalPathOptimizationNode:
    def __init__(self):
        rospy.init_node('global_path_optimization_node')
        
        # Roboterparameter
        self.robot_radius = rospy.get_param('~robot_radius', 0.5)
        self.max_steering_angle = rospy.get_param('~max_steering_angle', np.pi / 4)
        self.smoothing_factor = rospy.get_param('~smoothing_factor', 10)

        # Abonnieren des globalen Pfads
        self.global_path_subscriber = rospy.Subscriber('/move_base_flex/GlobalPlanner/plan', Path, self.global_path_callback)
        # Veröffentlichen des optimierten Pfads
        self.optimized_path_publisher = rospy.Publisher('/optimized_path', Path, queue_size=10)

        rospy.sleep(10) 

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

        # Anzeigen des optimierten Pfads mit Matplotlib
        self.plot_path(smoothed_path)

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
        # Vorbereitung der Daten für die Spline-Interpolation
        path = np.array(path)
        s_value = len(path) * self.smoothing_factor
        tck, _ = splprep(path.T, k=3, s=s_value)

        # Neu parametrisieren des Pfads
        u_new = np.linspace(0, 1, s_value)
        x_new, y_new = splev(u_new, tck)

        # Zusammensetzen des neuen Pfads
        smoothed_path = np.column_stack((x_new, y_new))

        return smoothed_path

    
    def plot_path(self, path):
        # Funktion zum Plotten eines Pfads mit Matplotlib
        plt.figure()
        plt.plot(path[:,0], path[:,1], 'b-', linewidth=2)
        plt.title('Optimized Path')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.axis('equal')


if __name__ == '__main__':
    try:
        GlobalPathOptimizationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
