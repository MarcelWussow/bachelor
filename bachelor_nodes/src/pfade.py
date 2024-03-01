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

        # Anzahl der alternativen Pfade
        self.num_alternative_paths = rospy.get_param('~num_alternative_paths', 100)
        # Glättungsparameter s
        self.smoothing_factor = rospy.get_param('~smoothing_factor_pfade', 10)

        self.optimized_path_subscriber = rospy.Subscriber('/optimized_path', Path, self.optimized_path_callback)
        self.map_subscriber = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.path_publisher = rospy.Publisher('/best_path', Path, queue_size=10)

        rospy.sleep(10) 

        # Globale Variable für die Hinderniskarte
        self.occupancy_grid = None

        # Hüllkurvenpunkte und Form 
        self.robot_hull_points = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])   #Rechteck 2x2 m

        # Hüllkurvenpunkte des Roboters
        self.robot_hull = self.generate_robot_hull()

    def optimized_path_callback(self, path_msg):
        # 100 alternative Pfade erzeugen
        alternative_paths = self.generate_alternative_paths(path_msg)

        # Überprüfen, ob die Pfade gültig sind und Bewertung
        best_path = path_msg
        best_evaluation = self.evaluate_path(path_msg)
        rospy.loginfo("Optimized path evaluation: {}".format(best_evaluation))

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

            # Matplotlib-Plot der globalen Hindernisse hinzugefügt
            self.plot_global_obstacles(self.occupancy_grid)
            
    def map_callback(self, map_msg):
        # Aktualisieren der Hinderniskarte
        self.occupancy_grid = map_msg

    def generate_robot_hull(self):
        # Hüllkurvenpunkte des Roboters 
        robot_points = np.array(self.robot_hull_points)
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
            if self.point_within_hull(point) and self.point_not_in_obstacle(point):
                random_points.append(point)

        return np.array(random_points)

    def point_within_hull(self, point):
        # Überprüfen, ob ein Punkt innerhalb der Hüllkurve liegt
        return all(
            np.dot(eq[:-1], point) + eq[-1] <= 0
            for eq in self.robot_hull.equations
        )

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
    
    def hull_not_collide_with_obstacle(self, path):
        # Überprüfen, ob die Hüllkurve entlang des Pfads nicht mit Hindernissen kollidiert
        for point in path:
            if not self.point_not_in_obstacle(point):
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
        # Überprüfen, ob ein Pfad gültig ist (nicht in einem Hindernis, Mindestabstand zu Hindernissen und Hüllkurve nicht kollidiert mit Hindernissen)
        for point in path:
            if not self.point_within_hull(point) or not self.point_not_in_obstacle(point) or not self.min_distance_to_obstacle(point) or not self.hull_not_collide_with_obstacle(path):
                return False
        return True

    def evaluate_path(self, path):
        # Pfadbewertung (Lenkwinkeländerungen bestrafen, Geradeausfahrten positiv bewerten, Pfadlänge bestrafen)
        steering_penalty = np.sum(np.abs(np.diff(path, axis=0)))
        straight_reward = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        path_length_penalty = len(path)

        evaluation = straight_reward - steering_penalty - path_length_penalty
        
        return evaluation

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

    def plot_global_obstacles(self, occupancy_grid):
        # Matplotlib-Plot der globalen Hindernisse
        if occupancy_grid is not None:
            resolution = occupancy_grid.info.resolution
            origin_x = occupancy_grid.info.origin.position.x
            origin_y = occupancy_grid.info.origin.position.y
            width = occupancy_grid.info.width
            height = occupancy_grid.info.height
            occupancy_data = np.array(occupancy_grid.data).reshape((width, height))
            grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
            obstacles_x = origin_x + grid_x * resolution
            obstacles_y = origin_y + grid_y * resolution
            obstacles = np.stack((obstacles_x, obstacles_y), axis=-1)
            obstacles = obstacles[occupancy_data > 50]  # Annahme: Werte über 50 sind Hindernisse
            plt.plot(obstacles[:, 0], obstacles[:, 1], 'k.', markersize=1)

    def plot_hull_along_path(self, path, title):
        # Funktion zum Plotten der Hüllkurve entlang des Pfads mit Matplotlib
        plt.figure()
        plt.plot(path[:, 0], path[:, 1], 'b-', linewidth=2)
        plt.plot(path[:, 0], path[:, 1], 'bo')  # Punkte des Pfads markieren
        hull_points = self.generate_robot_hull_along_path(path)
        plt.plot(hull_points[:, 0], hull_points[:, 1], 'r--', linewidth=2)
        plt.plot(hull_points[:, 0], hull_points[:, 1], 'ro')  # Punkte der Hüllkurve markieren
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    def generate_robot_hull_along_path(self, path):
        # Generiere die Hüllkurve entlang des Pfads
        hull_points = []
        for point in path:
            # Berechne die Hüllkurve für den Punkt
            robot_hull = ConvexHull(self.robot_hull_points + point)
            hull_points.append(self.robot_hull_points[robot_hull.vertices] + point)
        return np.concatenate(hull_points)

if __name__ == '__main__':
    try:
        path_evaluator = PathEvaluator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
