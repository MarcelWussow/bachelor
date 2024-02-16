#!/usr/bin/env python3
import rospy #type:ignore
from nav_msgs.msg import Path #type:ignore
from geometry_msgs.msg import PoseStamped, Point #type:ignore
import math #type:ignore

class PathOptimizerNode:
    def __init__(self):
        rospy.init_node('path_optimizer_node', anonymous=True)
        
        # Subscriber für den globalen Pfad
        rospy.Subscriber('/global_planner/path', Path, self.path_callback)
        
        # Publisher für den optimierten Pfad
        self.optimized_path_pub = rospy.Publisher('/optimized_path', Path, queue_size=10)
        
        # Publisher für den Pfad, der in RViz angezeigt wird
        self.rviz_path_pub = rospy.Publisher('/rviz/path', Path, queue_size=10)
        
        rospy.spin()

    def path_callback(self, path_msg):
        # Hier implementiere den Algorithmus zur Optimierung des Pfads
        optimized_path_msg = self.optimize_path(path_msg)
        
        # Veröffentliche den optimierten Pfad
        self.optimized_path_pub.publish(optimized_path_msg)
        
        # Veröffentliche den Pfad für RViz
        self.rviz_path_pub.publish(path_msg)

    def optimize_path(self, path_msg):
        optimized_path_msg = Path()
        optimized_path_msg.header = path_msg.header

        # Parameter für die Geometrie des Roboters
        L = 0.5  # Beispiel: Länge zwischen den Rädern (Radstand)
        theta_max = math.radians(30)  # Beispiel: Maximaler Lenkwinkel in Radian
        max_orientation_error = math.radians(10)  # Maximaler Orientierungsfehler in Radian
        
        # Anzahl der Zwischenpunkte für die Bezier-Kurve
        num_points = 10

        for i in range(len(path_msg.poses) - 1):
            pose_start = path_msg.poses[i].pose
            pose_end = path_msg.poses[i+1].pose

            # Berechne die Orientierung des Roboters zwischen den aktuellen und nächsten Positionen im Pfad
            orientation = math.atan2(pose_end.position.y - pose_start.position.y,
                                     pose_end.position.x - pose_start.position.x)
            
            # Überprüfe den Orientierungsfehler und passe die Orientierung des Roboters an, falls erforderlich
            if abs(orientation - pose_start.orientation.z) > max_orientation_error:
                pose_start.orientation.z = orientation
            
            # Konvertiere Start- und Endpunkte in Bezier-Form
            start_point = Point(pose_start.position.x, pose_start.position.y, 0)
            end_point = Point(pose_end.position.x, pose_end.position.y, 0)

            # Berechne die Zwischenpunkte der Bezier-Kurve
            bezier_points = self.calculate_bezier_points(start_point, end_point, num_points)

            # Füge die Zwischenpunkte zur optimierten Pfadnachricht hinzu
            for point in bezier_points:
                pose_stamped = PoseStamped()
                pose_stamped.pose.position = point

                # Berechne den Abstand zwischen den Punkten
                dx = point.x - pose_start.position.x
                dy = point.y - pose_start.position.y
                distance = math.sqrt(dx**2 + dy**2)

                # Berechne den maximalen Versatz des Roboters
                x_max = L / math.tan(theta_max)

                # Berechne den minimalen fahrbaren Radius
                min_radius = L / 2 + math.sqrt((L / 2)**2 + (distance - x_max)**2)

                # Speichere den minimalen fahrbaren Radius im 'z'-Feld der Pose
                pose_stamped.pose.position.z = min_radius
                
                optimized_path_msg.poses.append(pose_stamped)

        return optimized_path_msg

    def calculate_bezier_points(self, start_point, end_point, num_points):
        # Beispielhafte Bezier-Kurvenberechnung: Linear interpolieren zwischen Start- und Endpunkt
        bezier_points = []
        for t in range(num_points):
            x = start_point.x + (end_point.x - start_point.x) * t / (num_points - 1)
            y = start_point.y + (end_point.y - start_point.y) * t / (num_points - 1)
            bezier_points.append(Point(x, y, 0))
        return bezier_points

if __name__ == '__main__':
    try:
        PathOptimizerNode()
    except rospy.ROSInterruptException:
        pass