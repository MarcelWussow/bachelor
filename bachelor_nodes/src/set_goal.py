#!/usr/bin/env python3
import rospy #type:ignore
from geometry_msgs.msg import PoseStamped #type:ignore

def main():
    rospy.init_node("map_navigation_publisher", anonymous=False)

    x_goal = rospy.get_param("~x_goal", 1)
    y_goal = rospy.get_param("~y_goal", 1)

    goal_publisher = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)

    rospy.sleep(5) 

    goal_msg = PoseStamped()
    goal_msg.header.frame_id = "map"
    goal_msg.header.stamp = rospy.Time.now()

    goal_msg.pose.position.x = x_goal
    goal_msg.pose.position.y = y_goal
    goal_msg.pose.position.z = 0.0

    goal_msg.pose.orientation.x = 0.0
    goal_msg.pose.orientation.y = 0.0
    goal_msg.pose.orientation.z = 0.0
    goal_msg.pose.orientation.w = 1.0

    rospy.loginfo(f"Sending goal location: ({x_goal}, {y_goal})")
    goal_publisher.publish(goal_msg)
    
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass