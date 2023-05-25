#! /usr/bin/env python

"""
Standalone script for testing head movement.
"""

from __future__ import print_function

import sys
import rospy

# Brings in the SimpleActionClient
from actionlib import SimpleActionClient

# Brings in the messages used by the fibonacci action, including the
# goal message and the result message.
import actionlib_tutorials.msg

from control_msgs.msg import FollowJointTrajectoryAction
from control_msgs.msg import FollowJointTrajectoryGoal

from trajectory_msgs.msg import JointTrajectoryPoint

def client():
    # Creates the SimpleActionClient, passing the type of the action
    client = SimpleActionClient('/head_controller/follow_joint_trajectory', FollowJointTrajectoryAction)

    # Waits until the action server has started up and started
    # listening for goals.
    client.wait_for_server()

    rospy.loginfo('Server connected')

    # Creates a goal to send to the action server.
    goal = FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = ['head_1_joint', 'head_2_joint']
    goal.trajectory.points.append(JointTrajectoryPoint(
            positions=[0.2, 0.2],
            velocities=[0.0, 0.0],
            accelerations=[0.0, 0.0],
            effort=[0.0, 0.0]))
    goal.trajectory.points[-1].time_from_start.secs = 3

    # Sends the goal to the action server.
    client.send_goal(goal)

    # Waits for the server to finish performing the action.
    client.wait_for_result()

    # Prints out the result of executing the action
    return client.get_result()  # A FibonacciResult

if __name__ == '__main__':
    try:
        # Initializes a rospy node so that the SimpleActionClient can
        # publish and subscribe over ROS.
        rospy.init_node('head_client_py')
        result = client()
        print("Result:", result)
    except rospy.ROSInterruptException:
        print("program interrupted before completion", file=sys.stderr)

