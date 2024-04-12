#!/usr/bin/env python3
import rospy #type:ignore
from nav_msgs.msg import Path #type:ignore
import matplotlib.pyplot as plt #type:ignore
from math import sqrt #type:ignore

class PathSmoothingNode:
    def __init__(self):
        rospy.init_node('path_smoothing_node')

        # Abonniere den globalen Pfad und veröffentliche den optimierten und geglätteten Pfad
        self.global_path_subscriber = rospy.Subscriber('/move_base_flex/GlobalPlanner/plan', Path, self.global_path_callback)
        self.direct_path_publisher = rospy.Publisher('/direct_path', Path, queue_size=10)

    def global_path_callback(self, path):
        douglas_peucker = DouglasPeucker(epsilon=0.5)  # Setzen Sie hier das gewünschte Epsilon
        simplified_path_poses = douglas_peucker.simplify(path.poses)
        simplified_path_msg = Path(header=path.header, poses=simplified_path_poses)
        self.direct_path_publisher.publish(simplified_path_msg)
        self.plot_paths(path, simplified_path_msg)


    def plot_paths(self, original_path, direct_path):
        original_x = [pose.pose.position.x for pose in original_path.poses]
        original_y = [pose.pose.position.y for pose in original_path.poses]

        direct_x = [pose.pose.position.x for pose in direct_path.poses]
        direct_y = [pose.pose.position.y for pose in direct_path.poses]

        plt.figure()
        plt.plot(original_x, original_y, label='Original Path')
        plt.plot(direct_x, direct_y, label='Direct Path')
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
        # Berechnen Sie die Entfernung eines Punktes p zu einer Linie, die von p1 zu p2 verläuft
        x1, y1 = p1.pose.position.x, p1.pose.position.y
        x2, y2 = p2.pose.position.x, p2.pose.position.y
        x, y = p.pose.position.x, p.pose.position.y

        #Betrag Kreuzprodukt der Vektoren p1->p2 und p1->p (Flächeninhalt Parallelogramm) Satz von Gauß
        numer = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) 
        #Länge des Vektors p1->p2 Pythagoras
        denom = sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) 
        return numer / denom #Abstand P zur Linie p1->p2, ergibt den Abstand d

    def simplify(self, point_list):
        # Implementierung des Douglas-Peucker-Algorithmus auf die Punktliste
        if len(point_list) < 3: 
            return point_list

        # Finden des Punktes mit der größten Entfernung
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

        # Überprüfen ob die maximale Entfernung größer als das Epsilon ist
        if dmax > self.epsilon:
            # Rekursiv den Algorithmus auf den beiden Teilsegmenten anwenden
            left_segment = self.simplify(point_list[:index+1])
            right_segment = self.simplify(point_list[index:])
            return left_segment[:-1] + right_segment  # Entfernen des doppelten Endpunktes
        else:
            # Wenn die maximale Entfernung kleiner oder gleich dem Epsilon ist,
            # den Anfangs- und Endpunkt zurückgeben
            return [start_point, end_point]



if __name__ == '__main__':
    try:
        path_smoothing_node = PathSmoothingNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        rospy.signal_shutdown('Shutdown')