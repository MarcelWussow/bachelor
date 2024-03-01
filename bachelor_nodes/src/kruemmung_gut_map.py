#!/usr/bin/env python3
import rospy #type:ignore
from nav_msgs.msg import Path, OccupancyGrid #type:ignore
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
        # Abonnieren der Hinderniskarte
        self.map_subscriber = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        # Veröffentlichen des optimierten Pfads
        self.optimized_path_publisher = rospy.Publisher('/optimized_path', Path, queue_size=10)

        self.occupancy_grid = None
        rospy.sleep(5) 

    def map_callback(self, map_msg):
        # Aktualisierung der Hinderniskarte
        self.occupancy_grid = map_msg

    def global_path_callback(self, path_msg):
        path = np.array([(pose.pose.position.x, pose.pose.position.y) for pose in path_msg.poses])

        # Identifizierung von Kurvenabschnitten im Pfad
        curve_indices = self.identify_curve_sections(path)

        # Anpassung von Kurven mit Dubin-Modell
        adjusted_path = self.adjust_curves_with_dubins(path, curve_indices)

        # Glättung des Pfads basierend auf C1-Stetigkeit und Orientierung
        smoothed_path = self.ensure_c1_continuity(adjusted_path)

        # Veröffentlichen des optimierten Pfads
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
            if abs(angle) > self.curve_threshold:
                curve_indices.append(i)
        return curve_indices
    
    def calculate_curvature_angle(self, p0, p1, p2):
        v1 = np.array(p1) - np.array(p0)
        v2 = np.array(p2) - np.array(p1)
        
        # Nullvektor ausschließen
        if np.all(v1 == 0) or np.all(v2 == 0):
            return 0.0
        
        # Berechnung des Skalarprodukts und der Normen
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        # Überprüfen, ob der Ausdruck innerhalb des Wertebereichs von -1 bis 1 liegt
        if norm_v1 * norm_v2 < 1e-6:  # Division durch Null vermeiden
            return 0.0
        else:
            cos_angle = dot_product / (norm_v1 * norm_v2)
            cos_angle = max(min(cos_angle, 1.0), -1.0)  # Wertebereich einschränken, um Ungenauigkeiten zu vermeiden
            angle = np.arccos(cos_angle)
            return angle

    def adjust_curves_with_dubins(self, path, curve_indices):
        adjusted_path = path.copy()
        for i in curve_indices:
            if i > 0 and i < len(path) - 1:
                p0, p1, p2 = path[i-1:i+2]

            # Finden des nächsten Hindernisses und Anpassen der Kurve
            nearest_obstacle = self.find_nearest_obstacle(p1)
            if nearest_obstacle is not None:  # Änderung hier
                adjusted_curve_point = self.adjust_point_to_radius(p0, p1, p2, nearest_obstacle)
                adjusted_path[i] = adjusted_curve_point
        return adjusted_path

    def find_nearest_obstacle(self, point):
        if self.occupancy_grid is None:
            return None
        
        width = self.occupancy_grid.info.width
        height = self.occupancy_grid.info.height
        resolution = self.occupancy_grid.info.resolution
        origin_x = self.occupancy_grid.info.origin.position.x
        origin_y = self.occupancy_grid.info.origin.position.y
        data = self.occupancy_grid.data

        map_x = int((point[0] - origin_x) / resolution)
        map_y = int((point[1] - origin_y) / resolution)

        search_radius = int(self.robot_radius / resolution)
        min_dist = float('inf')
        nearest_obstacle = None

        for x in range(max(map_x - search_radius, 0), min(map_x + search_radius, width)):
            for y in range(max(map_y - search_radius, 0), min(map_y + search_radius, height)):
                if data[y * width + x] > 0:
                    obstacle_x = x * resolution + origin_x
                    obstacle_y = y * resolution + origin_y
                    dist = np.linalg.norm(np.array([obstacle_x, obstacle_y]) - np.array(point))
                    if dist < min_dist:
                        min_dist = dist
                        nearest_obstacle = np.array([obstacle_x, obstacle_y])

        return nearest_obstacle

    def adjust_point_to_radius(self, p0, p1, p2, obstacle):
        # Berechnen der Richtungsvektoren
        v1 = np.array(p1) - np.array(p0)
        v2 = np.array(p2) - np.array(p1)

        # Normalisierung von v2
        v2_norm = v2 / np.linalg.norm(v2)

        # Berechnung des Richtungsvektors basierend auf v1 und v2
        direction_vector = np.array([-v1[1], v1[0]])  # Senkrechter Vektor zu v1

        # Berechnung des Mittelpunkts des Bogens
        mid_point = np.array(p1) + 0.5 * v2_norm * np.linalg.norm(v2)

        # Berechnung des adjustierten Punktes entlang der Tangente
        adjusted_point = mid_point + direction_vector * self.robot_radius

        return adjusted_point
    
    def ensure_c1_continuity(self, path):
        path = np.array(path)
        s_value = len(path) * self.smoothing_factor
        tck, _ = splprep(path.T, k=3, s=s_value)

        u_new = np.linspace(0, 1, s_value)
        x_new, y_new = splev(u_new, tck)

        smoothed_path = np.column_stack((x_new, y_new))

        return smoothed_path
    
    def plot_path(self, path):
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
