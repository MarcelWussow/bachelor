#!/usr/bin/env python3
import rospy #type:ignore
from nav_msgs.msg import Path #type:ignore
import numpy as np #type:ignore
import matplotlib.pyplot as plt #type:ignore

class PathCurvatureAnalyzer:
    def __init__(self):
        self.path_sub = rospy.Subscriber('/global_planner/path', Path, self.path_callback)

    def path_callback(self, path_msg):
        # Extrahiere die Punkte aus der Pfadnachricht
        path = [(pose.pose.position.x, pose.pose.position.y) for pose in path_msg.poses]

        # Berechne die Krümmung basierend auf drei aufeinanderfolgenden Punkten im Pfad
        curvature = []
        for i in range(1, len(path)-1):
            x_prev, y_prev = path[i-1]
            x_curr, y_curr = path[i]
            x_next, y_next = path[i+1]

            # Berechne die Winkeländerung zwischen den Vektoren der aufeinanderfolgenden Punkte
            angle_change = np.abs(np.arctan2(y_next - y_curr, x_next - x_curr) - np.arctan2(y_curr - y_prev, x_curr - x_prev))
            curvature.append(angle_change)

        # Visualisiere die Krümmung des Pfads
        plt.plot(range(len(curvature)), curvature)
        plt.xlabel('Position im Pfad')
        plt.ylabel('Krümmung')
        plt.title('Krümmung des Pfads')
        plt.show()

if __name__ == '__main__':
    rospy.init_node('path_curvature_analyzer')
    path_analyzer = PathCurvatureAnalyzer()
    rospy.spin()

# noch raussuchen für welchen winkel der pfad besondere aufmerksamkeit benötigt