#!/usr/bin/env python3
import rospy #type:ignore
from nav_msgs.msg import Path #type:ignore
from nav_msgs.msg import OccupancyGrid #type:ignore
import numpy as np #type:ignore
from scipy.spatial import ConvexHull #type:ignore
from scipy.interpolate import splprep, splev #type:ignore
import matplotlib.pyplot as plt #type:ignore

class PathEvaluator:
    def __init__(self):
        rospy.init_node('path_evaluator')
        self.optimized_path_subscriber = rospy.Subscriber('/optimized_path', Path, self.optimized_path_callback)
        self.map_subscriber = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.path_publisher = rospy.Publisher('/best_path', Path, queue_size=10)

        rospy.sleep(15) 

        # Hüllkurvenpunkte des Roboters
        self.robot_hull = self.generate_robot_hull()

        # Globale Variable für die Hinderniskarte
        self.occupancy_grid = None

        # Hüllkurvenpunkte und Form 
        self.hull_points_param = rospy.get_param('~robot_hull_points')

        # Anzahl der alternativen Pfade
        self.num_alternative_paths = rospy.get_param('~num_alternative_paths', 100)
        
        # Glättungsparameter s
        self.smoothing_factor = rospy.get_param('~smoothing_factor_pfade', 10)

    def optimized_path_callback(self, path_msg):
        # 100 alternative Pfade erzeugen
        alternative_paths = self.generate_alternative_paths(path_msg)

        # Überprüfen, ob die Pfade gültig sind und Bewertung
        best_path = None
        best_evaluation = float('-inf')
        for path in alternative_paths:
            if self.is_valid_path(path):
                evaluation = self.evaluate_path(path)
                rospy.loginfo("Path evaluation: {}".format(evaluation))
                if evaluation > best_evaluation:
                    best_evaluation = evaluation
                    best_path = path

        rospy.loginfo("Best path evaluation: {}".format(best_evaluation))
        rospy.loginfo("Best path: {}".format(best_path))

        if best_path is not None:
            self.path_publisher.publish(best_path)
            
            # Matplotlib-Plot des besten Pfads hinzugefügt
            self.plot_path(best_path, 'Best Path')

            # Matplotlib-Plot der alternativen Pfade hinzugefügt
            for i, path in enumerate(alternative_paths):
                self.plot_path(path, f'Alternative Path {i+1}')

            # Matplotlib-Plot der Roboterhülle hinzugefügt
            self.plot_hull(self.robot_hull.points, 'Robot Hull')

    def map_callback(self, map_msg):
        # Aktualisieren der Hinderniskarte
        self.occupancy_grid = map_msg

    def generate_robot_hull(self):
        # Hüllkurvenpunkte des Roboters aus den rosparam lesen
        robot_points = np.array(self.hull_points_param)
        robot_hull = ConvexHull(robot_points)
        return robot_hull

    def generate_alternative_paths(self, path_msg):
        alternative_paths = []
        for _ in range(self.num_alternative_paths):
            # Zufällige Punkte innerhalb der Hüllkurve generieren
            random_points = self.generate_random_points_within_hull()

            # Pfad durch Glättung der zufälligen Punkte erhalten
            smoothed_path = self.smooth_path(random_points)

            alternative_paths.append(smoothed_path)

        return alternative_paths

    def generate_random_points_within_hull(self):
        # Zufällige Punkte innerhalb der Hüllkurve generieren
        min_x, max_x = self.robot_hull.points[:, 0].min(), self.robot_hull.points[:, 0].max()
        min_y, max_y = self.robot_hull.points[:, 1].min(), self.robot_hull.points[:, 1].max()

        random_points = []
        while len(random_points) < 10:  # Anzahl der benötigten Punkte
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            point = np.array([x, y])
            if self.point_within_hull(point) and self.point_not_in_obstacle(point) and self.min_distance_to_obstacle(point):
                random_points.append(point)

        return np.array(random_points)

    def point_within_hull(self, point):
        # Überprüfen, ob ein Punkt innerhalb der Hüllkurve liegt
        return self.robot_hull.find_simplex(point) >= 0

    def point_not_in_obstacle(self, point):
        # Überprüfen, ob ein Punkt nicht in einem Hindernis liegt
        if self.occupancy_grid is not None:
            resolution = self.occupancy_grid.info.resolution
            origin_x = self.occupancy_grid.info.origin.position.x
            origin_y = self.occupancy_grid.info.origin.position.y
            width = self.occupancy_grid.info.width
            occupancy_data = np.array(self.occupancy_grid.data).reshape((width, width))
            map_x = int((point[0] - origin_x) / resolution)
            map_y = int((point[1] - origin_y) / resolution)
            if 0 <= map_x < width and 0 <= map_y < width:
                return occupancy_data[map_x, map_y] <= 50  # Annahme: 0-50 sind freier Raum, Werte darüber sind Hindernisse

        return True

    def min_distance_to_obstacle(self, point):
        # Überprüfen, ob ein Punkt mindestens 0,5m Abstand zu Hindernissen hat
        if self.occupancy_grid is not None:
            resolution = self.occupancy_grid.info.resolution
            origin_x = self.occupancy_grid.info.origin.position.x
            origin_y = self.occupancy_grid.info.origin.position.y
            width = self.occupancy_grid.info.width
            occupancy_data = np.array(self.occupancy_grid.data).reshape((width, width))
            map_x = int((point[0] - origin_x) / resolution)
            map_y = int((point[1] - origin_y) / resolution)
            if 0 <= map_x < width and 0 <= map_y < width:
                if occupancy_data[map_x, map_y] > 50:  # Wenn Punkt in Hindernis liegt
                    return False
                # Überprüfen des Abstands zu Hindernissen
                for i in range(-5, 6):  # Überprüfen in einem 1m x 1m Bereich um den Punkt
                    for j in range(-5, 6):
                        if (map_x + i) >= 0 and (map_x + i) < width and (map_y + j) >= 0 and (map_y + j) < width:
                            if occupancy_data[map_x + i, map_y + j] > 50:
                                distance = np.sqrt((i * resolution) ** 2 + (j * resolution) ** 2)
                                if distance < 0.5:
                                    return False
        return True

    def smooth_path(self, points):
        # Pfadglättung mit kubischen B-Splines
        points = np.array(points)
        tck, _ = splprep(points.T, k=3, s=self.smoothing_factor)
        u_new = np.linspace(0, 1, len(points) * 10)
        smoothed_path = splev(u_new, tck)
        smoothed_path = np.array(smoothed_path).T
        return smoothed_path

    def is_valid_path(self, path):
        # Überprüfen, ob ein Pfad gültig ist (z.B. innerhalb der Hüllkurve, nicht in einem Hindernis und Mindestabstand zu Hindernissen)
        for point in path:
            if not self.point_within_hull(point) or not self.point_not_in_obstacle(point) or not self.min_distance_to_obstacle(point):
                return False
        return True

    def evaluate_path(self, path):
        # Pfadbewertung (Lenkwinkeländerungen bestrafen, Geradeausfahrten positiv bewerten, Pfadlänge bestrafen)
        steering_penalty = np.sum(np.abs(np.diff(path, axis=0)))
        straight_reward = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        path_length_penalty = len(path)

        evaluation = straight_reward - steering_penalty - path_length_penalty
        
        return evaluation

    def nearest_obstacle_distance(self, x, y):
        # Finden des nächsten Hindernisses zu einem bestimmten Punkt
        if self.occupancy_grid is not None:
            resolution = self.occupancy_grid.info.resolution
            origin_x = self.occupancy_grid.info.origin.position.x
            origin_y = self.occupancy_grid.info.origin.position.y
            width = self.occupancy_grid.info.width
            occupancy_data = np.array(self.occupancy_grid.data).reshape((width, width))
            map_x = int((x - origin_x) / resolution)
            map_y = int((y - origin_y) / resolution)
            if 0 <= map_x < width and 0 <= map_y < width:
                if occupancy_data[map_x, map_y] > 50:  # Wenn Punkt in Hindernis liegt
                    return 0.0
                min_distance = float('inf')
                for i in range(-5, 6):  # Überprüfen in einem 1m x 1m Bereich um den Punkt
                    for j in range(-5, 6):
                        if (map_x + i) >= 0 and (map_x + i) < width and (map_y + j) >= 0 and (map_y + j) < width:
                            if occupancy_data[map_x + i, map_y + j] > 50:
                                distance = np.sqrt((i * resolution) ** 2 + (j * resolution) ** 2)
                                if distance < min_distance:
                                    min_distance = distance
                return min_distance
        return None

    def plot_path(self, path, title):
        # Funktion zum Plotten eines Pfads mit Matplotlib
        plt.figure()
        plt.plot(path[:, 0], path[:, 1], 'b-', linewidth=2)
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.axis('equal')

    def plot_hull(self, points, title):
        # Funktion zum Plotten einer Hüllkurve mit Matplotlib
        plt.figure()
        plt.plot(points[:, 0], points[:, 1], 'r--', linewidth=2)
        plt.plot(points[:, 0], points[:, 1], 'ro')  # Punkte der Hüllkurve markieren
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.axis('equal')


if __name__ == '__main__':
    try:
        path_evaluator = PathEvaluator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
