#!/usr/bin/env python3
import rosbag #type:ignore
import matplotlib.pyplot as plt #type:ignore
import numpy as np #type:ignore

def plot_map(map_msg):
    # Plot der Karte
    plt.figure()
    map_data = np.array(map_msg.data).reshape((map_msg.info.width, map_msg.info.height))
    plt.imshow(map_data, cmap='gray')
    plt.title('Occupancy Grid Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar()
    plt.show()

def plot_path(path_msg, title):
    # Plot des Pfads
    plt.figure()
    poses = [(pose.pose.position.x, pose.pose.position.y) for pose in path_msg.poses]
    x = [pose[0] for pose in poses]
    y = [pose[1] for pose in poses]
    plt.plot(x, y, 'b-', linewidth=2)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

bag_file = 'deine_aufnahme.bag'  # Passe den Dateinamen entsprechend an

# Öffne die rosbag-Datei
with rosbag.Bag(bag_file, 'r') as bag:
    for topic, msg, t in bag.read_messages():
        if topic == '/map':
            plot_map(msg)
        elif topic == '/global_planner/path':
            plot_path(msg, 'Global Planner Path')
        elif topic == '/optimized_path':
            plot_path(msg, 'Optimized Path')
        elif topic == '/best_path':
            plot_path(msg, 'Best Path')