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
        self.steering_angle_deg = rospy.get_param('~steering_angle_deg', 45.0)  # Lenkwinkel in Grad
        self.smoothing_factor = rospy.get_param('~smoothing_factor', 10)
        self.curve_threshold_deg = rospy.get_param('~curve_threshold_deg', 5.0)  # Winkel für die Beurteilung der Krümmung in Grad

        # Konvertiere die Euler-Winkel von Grad nach Radian
        self.max_steering_angle = np.deg2rad(self.steering_angle_deg)
        self.curve_threshold = np.deg2rad(self.curve_threshold_deg)

        # Abonnieren des globalen Pfads
        self.global_path_subscriber = rospy.Subscriber('/move_base_flex/GlobalPlanner/plan', Path, self.global_path_callback)
        # Veröffentlichen des optimierten Pfads
        self.optimized_path_publisher = rospy.Publisher('/optimized_path', Path, queue_size=10)

        rospy.sleep(5) 

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
            if abs(angle) > self.curve_threshold:  # Verwendung des Schwellenwerts für Krümmungswinkel
                curve_indices.append(i)
        return curve_indices
    
    def calculate_curvature_angle(self, p0, p1, p2):
        v1 = np.array(p1) - np.array(p0)
        v2 = np.array(p2) - np.array(p1)
        
        # Überprüfen, ob v1 und v2 Nullvektoren sind
        if np.all(v1 == 0) or np.all(v2 == 0):
            return 0.0  # Rückgabe eines Standardwerts, wenn v1 oder v2 Nullvektoren sind
        
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        return angle

    def adjust_curves_with_dubins(self, path, curve_indices):
        adjusted_path = path.copy()
        for i in curve_indices:
            if i > 0 and i < len(path) - 1:
                p0, p1, p2 = path[i-1:i+2]
                adjusted_curve_point = self.calculate_smoothed_curve_point(p0, p1, p2)
                adjusted_path[i] = adjusted_curve_point
        return adjusted_path

    def calculate_smoothed_curve_point(self, p0, p1, p2):
        v1 = np.array(p1) - np.array(p0)
        v2 = np.array(p2) - np.array(p1)
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # Überprüfen, ob angle NaN ist oder distance_to_mid Null ist
        if np.isnan(angle) or np.isinf(angle) or np.isclose(angle, 0.0):
            # Behandlung des Falls, wenn angle NaN oder unendlich ist oder distance_to_mid Null ist
            return p1
        
        min_turning_radius = self.robot_radius / np.sin(min(angle, self.max_steering_angle))

        # Mittelpunkt des Kreises
        mid_point = np.array(p1)

        # Vektor vom Mittelpunkt zu einem der Endpunkte
        p1_mid_vector = mid_point - np.array(p1)

        # Projektion auf den Richtungsvektor von p1
        projection_length = np.dot(p1_mid_vector, v1)
        projected_point = np.array(p1) + projection_length * v1

        # Abstand zwischen dem Projektionspunkt und dem Mittelpunkt
        distance_to_mid = np.linalg.norm(mid_point - projected_point)

        # Überprüfen, ob distance_to_mid Null ist, um eine Division durch Null zu vermeiden
        if np.isclose(distance_to_mid, 0.0):
            return p1  # Rückgabe von p1, wenn distance_to_mid Null ist

        # Skalierungsfaktor für den minimalen fahrbaren Radius
        scale_factor = min_turning_radius / distance_to_mid

        # Berechnung des korrigierten Punktes
        corrected_point = mid_point + scale_factor * p1_mid_vector

        return corrected_point

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