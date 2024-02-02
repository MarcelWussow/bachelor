#!/usr/bin/env python3
import rospy #type:ignore
from geometry_msgs.msg import PoseStamped #type:ignore

def move_to_goal(xGoal, yGoal):
    goal_msg = PoseStamped()
    goal_msg.header.frame_id = "map"
    goal_msg.header.stamp = rospy.Time.now()

    goal_msg.pose.position.x = xGoal
    goal_msg.pose.position.y = yGoal
    goal_msg.pose.position.z = 0.0

    goal_msg.pose.orientation.x = 0.0
    goal_msg.pose.orientation.y = 0.0
    goal_msg.pose.orientation.z = 0.0
    goal_msg.pose.orientation.w = 1.0

    return goal_msg

#def check_goal_reached():
    # Simulating the condition when the robot has successfully reached the destination
    # Replace this with your actual logic or feedback mechanisms
    #if robot_has_reached_destination_condition:
        #rospy.loginfo("Robot has successfully reached the destination!")
        #return True
    #else:
        #rospy.loginfo("Robot failed to reach the destination.")
        #return False

def main():
    rospy.init_node("map_navigation_publisher", anonymous=False)
    goal_publisher = rospy.Publisher("goal_pose", PoseStamped, queue_size=10)

    x_goal = 7.5
    y_goal = 3

    rate = rospy.Rate(1)  # Publish rate in Hz

    while not rospy.is_shutdown():
        goal_msg = move_to_goal(x_goal, y_goal)
        rospy.loginfo("Sending goal location...")

        # Publish goal message
        goal_publisher.publish(goal_msg)
        rate.sleep()

        # Log information about goal achievement
        #rospy.loginfo("Checking if the robot has reached the destination...")
        #if check_goal_reached():
            #rospy.loginfo("Robot has successfully reached the destination!")
        #else:
            #rospy.loginfo("Robot failed to reach the destination.")

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
