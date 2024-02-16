#!/usr/bin/env python3
import rospy #type:ignore
from nav_msgs.msg import Path #type:ignore
import numpy as np #type:ignore
from scipy.interpolate import splprep, splev #type:ignore
import matplotlib.pyplot as plt #type:ignore

class PathSmoother:
    def __init__(self):
        self.path_sub = rospy.Subscriber('/global_planner/path', Path, self.path_callback)

    def path_callback(self, path_msg):
        # Extrahiere die x- und y-Koordinaten aus dem Pfad
        x_coords = [pose.pose.position.x for pose in path_msg.poses]
        y_coords = [pose.pose.position.y for pose in path_msg.poses]

        # Berechne kubische Splines für die x- und y-Koordinaten
        tck, _ = splprep([x_coords, y_coords], s=0)

        # Werte die Splines aus, um die geglätteten Koordinaten zu erhalten
        smoothed_x, smoothed_y = splev(np.linspace(0, 1, 100), tck)

        # Visualisiere den ursprünglichen Pfad und den geglätteten Pfad
        plt.plot(x_coords, y_coords, 'b-', label='Originaler Pfad')
        plt.plot(smoothed_x, smoothed_y, 'r--', label='Geglätteter Pfad')
        plt.xlabel('x-Koordinate')
        plt.ylabel('y-Koordinate')
        plt.title('Glättung des Pfads mit kubischen Splines')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    rospy.init_node('path_smoother')
    path_smoother = PathSmoother()
    rospy.spin()

#Dieser Code empfängt den abonnierten Pfad, extrahiert die x- und y-Koordinaten, berechnet kubische Splines 
#für diese Koordinaten und wertet sie aus, um die geglätteten x- und y-Koordinaten zu erhalten. Schließlich 
#visualisiert er den ursprünglichen Pfad und den geglätteten Pfad. Du kannst den Parameter s in splprep anpassen,
#um die Glättungseffekte zu steuern. Je größer der Wert von s, desto stärker wird der Pfad geglättet.