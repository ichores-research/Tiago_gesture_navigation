#!/usr/bin/env python
# license removed for brevity
# tento skript publikuje zpravy do topicu /arm_left_controller/command
# ovlada tak pouze testovaci levou ruku
# zaloha s funkcnim ovladanim pri prepisu souradnic rucne
import sys
import copy
import rospy
import time
import cv2
#import mediapipe as mp
#import numpy as np

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose, Quaternion
from tf.transformations import quaternion_from_euler
from math import pi, radians, degrees
from std_msgs.msg import String

from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryActionGoal, JointTolerance
from control_msgs.msg._FollowJointTrajectoryGoal import FollowJointTrajectoryGoal

from actionlib_msgs.msg._GoalID import GoalID
from actionlib.simple_action_client import SimpleActionClient

import moveit_msgs.msg
import moveit_commander
from moveit_commander.conversions import pose_to_list

ROBOT_JOINTS_MAX_VALUES_RADIANS = {
    "arm_left_1_joint": (-1.11, 1.50),
    "arm_left_2_joint": (-1.11, 1.50),
    "arm_left_3_joint": (-0.72, 3.86),
    "arm_left_4_joint": (-0.32, 2.29),
    "arm_left_5_joint": (-2.07, 2.07),
    "arm_left_6_joint": (-1.39, 1.39),
    "arm_left_7_joint": (-2.07, 2.07),
}

ROT_Y_CALIBRATION = -30
ROT_Y_STARTPOSE = 180
ROT_Z_STARTPOSE = 180

def transform_points_to_odom(final_point):
    """
    Function transforming location of human and the final point to odom coordinate system
    Input: 
        - final_point: Location3DJoint() object in MotionBERT human coordinate system
    Output: 
        - final_point_matrix: np.array() with coordinates of final point in odom coordinate system
        - human_coordinates: np.array() with coordinates of human in odom coordinate system
    """
    final_point_matrix = np.array([[final_point.x], [final_point.y], [1]])
    rot_matrix = get_rotation_matrix(-pi/2)
    final_point_matrix = np.matmul(rot_matrix, final_point_matrix)
    print("Bod po rotaci do zakladny cloveka")
    print(final_point_matrix)

    # Distance from robot base to human extracted from depth image
    distance_to_human = human_info.distance_from_robot
    # Human coordinates in human coordinate system -> [0, 0, 1]
    human_coordinates = np.array([[0], [0], [1]])
    trans_matrix = get_translation_matrix(distance_to_human, 0)
    final_point_matrix = np.matmul(trans_matrix, final_point_matrix)
    human_coordinates = np.matmul(trans_matrix, human_coordinates)
    print("Finalni bod po translaci do zakladny robota")
    print(final_point_matrix)
    print("Pozice cloveka po translaci do zakladny robota")
    print(human_coordinates)

    # Rotation to match the orientation of odom coordinate system.
    base_position_and_rotation = mobile_base.get_current_position_and_rotation()
    head_joint_position = latest_joint_states.return_head_joints_positions()
    trans_matrix = get_translation_matrix(base_position_and_rotation[0], base_position_and_rotation[1])
    rot_matrix = get_rotation_matrix(base_position_and_rotation[2] + head_joint_position[0])
    final_point_matrix = np.matmul(rot_matrix, final_point_matrix)
    human_coordinates = np.matmul(rot_matrix, human_coordinates)
    print("Finalni bod po rotaci do souradneho systemu odom")
    print(final_point_matrix)
    print("Pozice cloveka po rotaci do souradneho systemu odom")
    print(human_coordinates)

    # Translation to odom coordinate system.
    final_point_matrix = np.matmul(trans_matrix, final_point_matrix)
    human_coordinates = np.matmul(trans_matrix, human_coordinates)
    rospy.loginfo("Final point after transformation to odom coordinate system:")
    print(final_point_matrix)
    rospy.loginfo("Human position after transformation to odom coordinate system:")
    print(human_coordinates)
    return [final_point_matrix, human_coordinates]

def feedback_cb(msg):
    print("Zpetna vazba: %s" % msg)

def convert_image_to_video():
    """
    Funkce ktera z obrazku vytvori sekundove video pro zpracovani pomoci MotionBERT
    """
    image_folder = "/home/guest/image_recog_results/"
    video_name = 'human.mp4'

    images = ["human.jpg"]
    print(images)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(image_folder + video_name, fourcc, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    rospy.loginfo("Image converted to video...")

def wait_for_valid_time(timeout):
    """Wait for a valid time (non-zero), this is important
    when using a simulated clock"""
    # Loop until:
    # * ros master shutdowns
    # * control+C is pressed (handled in is_shutdown())
    # * timeout is achieved
    # * time is valid
    start_time = time.time()
    while not rospy.is_shutdown():
        if not rospy.Time.now().is_zero():
            return
        if time.time() - start_time > timeout:
            rospy.logerr("Timed-out waiting for valid time.")
            exit(0)
        time.sleep(0.1)
    # If control+C is pressed the loop breaks, we can exit
    exit(0)

def init_cartesian_control_for_group(group_name):
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("moveit_arm_left_test", anonymous=True)
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    #group_name = "arm_left_torso"
    group = moveit_commander.MoveGroupCommander(group_name)
    display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                               moveit_msgs.msg.DisplayTrajectory,
                                               queue_size=20)

    # We can get the name of the reference frame for this robot:
    planning_frame = group.get_planning_frame()
    print("============ Reference frame: %s" % planning_frame)

    # We can also print the name of the end-effector link for this group:
    eef_link = group.get_end_effector_link()
    print("============ End effector: %s" % eef_link)

    # We can get a list of all the groups in the robot:
    group_names = robot.get_group_names()
    print("============ Robot Groups:", group_names)

    # Sometimes for debugging it is useful to print the entire state of the
    # robot:
    print("============ Printing robot state")
    print(robot.get_current_state())
    print("")
    return group

def moveit_cartesinan_planner(group, coord_x, coord_y, coord_z, rot_x, rot_y, rot_z):
    """
    Funkce, ktera pomoci modulu Move IT! a inverzni kinematiky najde polohy kloubu pro zadanou polohu koncoveho efektoru robota
    group: skupina kloubu, ktera se ma pohybovat
    coord_x: x-ova souradnice koncoveho efektoru, kladny smer osy je kolmo pred robota
    coord_y: y_ova souradnice koncoveho efektrou, kladny smer je kolmo vlevo od robota
    coord_z: z-ova souradnice koncoveho efektrou, kladny smer je kolmo od zakladny robota
    polohove souradnice zadavane v metrech
    rot_x: otoceni kolem osy x, kladny smer po smeru hodin z pozice robota
    rot_y: otoceni kolem osy y, kladny smer zveda zapesti nahoru
    rot_z: otoceni kolem osy z, kladny smer naklani zapesti doleva
    """
    # definovani polohy, do ktere se ma umistit robot
    pose_goal = Pose()
    pose_goal.position.x = coord_x # 0.4 kladny smer kolmo pred robota
    pose_goal.position.y = coord_y # 0.3
    pose_goal.position.z = coord_z # 0.26

    # kladny smer rotaci je podle smeru hodin, postupne podle os xyz # pitch roll yaw
    orientation = quaternion_from_euler(radians(rot_x), radians(rot_y + ROT_Y_CALIBRATION + ROT_Y_STARTPOSE), radians(rot_z + ROT_Z_STARTPOSE))
    
    print(orientation)
    orientation_quat = Quaternion()
    orientation_quat.x = orientation[0] 
    orientation_quat.y = orientation[1] 
    orientation_quat.z = orientation[2] 
    orientation_quat.w = orientation[3]
    pose_goal.orientation = orientation_quat
    group.set_pose_target(pose_goal)

    # naplanuj a proved
    plan = group.go(wait=True)
    print(plan)
    group.stop()
    group.clear_pose_targets()

def hand_point_gesture():
    """
    funkce ktera ruku robota nastavi do tvaru ukazovani pomoci ukazovacku 
    """
    hand_action_client = SimpleActionClient('/hand_left_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    iterations = 0
    max_iterations = 3
    while(not hand_action_client.wait_for_server(rospy.Duration(5,0)) and iterations < max_iterations):
        rospy.loginfo("Creating action client to hand_left_controller ...")
        iterations+= 1

    if ( iterations == max_iterations ):
        rospy.loginfo("Action client not created!")
        exit(200)
    else:
        rospy.loginfo("Action client created")

    # vztvor trajektorii o jednom bode, ve kterem je ruka sevrena a ukazovacen naprimeny
    trajektorie = JointTrajectory()
    trajektorie.header.frame_id = "Sevreni leve ruky"
    trajektorie.header.seq += 1
    trajektorie.header.stamp = rospy.Time().now() + rospy.Duration(1,0)
    trajektorie.joint_names = ["hand_left_index_joint","hand_left_mrl_joint","hand_left_thumb_joint"]
    trajektorie.points = [JointTrajectoryPoint()]
    trajektorie.points[0].positions = [0.0, 2.5, 3.0]
    trajektorie.points[0].velocities = [0.0] * 3
    trajektorie.points[0].time_from_start = rospy.Duration(2,0)

    cil = FollowJointTrajectoryGoal()
    cil.trajectory = trajektorie
    cil.goal_time_tolerance = rospy.Duration(1,0)

    rospy.loginfo(cil)
    hand_action_client.send_goal(cil, feedback_cb=feedback_cb)
    hand_action_client.wait_for_result(rospy.Duration(10,0))

    result = hand_action_client.get_result()
    return result

def move_left_arm():
    """
    Funkce ktera podle zadanych hodnot positions nasmeruje klouby roboda do danzch uhlu v radianech
    """
    # vytvorim si noveho klienta pro ovladani leve ruky robota
    arm_action_client = SimpleActionClient('/arm_left_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    iterations = 0
    max_iterations = 3
    while(not arm_action_client.wait_for_server(rospy.Duration(5,0)) and iterations < max_iterations):
        rospy.loginfo("Creating action client to arm_left_controller ...")
        iterations+= 1

    if ( iterations == max_iterations ):
        rospy.loginfo("Action client not created!")
        exit(200)
    else:
        rospy.loginfo("Action client created")
    
    # definovani trajektorie dvou bodu
    trajektorie = JointTrajectory()
    trajektorie.header.frame_id = "Trajektorie z Pythonu"
    trajektorie.header.seq += 1
    trajektorie.header.stamp = rospy.Time().now() + rospy.Duration(1,0)
    trajektorie.joint_names = ["arm_left_1_joint","arm_left_2_joint","arm_left_3_joint","arm_left_4_joint","arm_left_5_joint","arm_left_6_joint","arm_left_7_joint"]
    trajektorie.points = [JointTrajectoryPoint(), JointTrajectoryPoint(), JointTrajectoryPoint(), JointTrajectoryPoint(), JointTrajectoryPoint(), JointTrajectoryPoint()]
    trajektorie.points[0].positions = [1.1, 0.59, 0.06, 1.0, -1.70, 0.0, 0.0] # [0.0] * 7 , [0.0, 0.59, 0.06, 1.0, -1.70, 0.0, 0.0]
    trajektorie.points[0].velocities = [0.0] * 7
    trajektorie.points[0].time_from_start = rospy.Duration(2,0)
    trajektorie.points[1].positions = [1.5, 0.3, -0.3, 1.0, -1.70, 0.0, 0.0] # [0.5] * 7 , [1.20, 0.59, 0.06, 1.0, -1.70, 0.0, 0.0]
    trajektorie.points[1].velocities = [0.0] * 7
    trajektorie.points[1].time_from_start = rospy.Duration(4,0)
    trajektorie.points[2].positions = [0.8, 0, 0.06, 0.45, -1.70, 0.0, 0.0] # [0.0] * 7, [0.1, 0.59, 0.06, 1.0, -1.70, 0.0, 0.0]
    trajektorie.points[2].velocities = [0.0] * 7
    trajektorie.points[2].time_from_start = rospy.Duration(6,0)
    trajektorie.points[3].positions = [0.1, 0.3, 0.3, 0.45, -1.70, 0.0, 0.0] # [0.5] * 7, [1.10, 0.59, 0.06, 1.0, -1.70, 0.0, 0.0]
    trajektorie.points[3].velocities = [0.0] * 7
    trajektorie.points[3].time_from_start = rospy.Duration(8,0)
    trajektorie.points[4].positions = [0.6, 0.5, 0.1, 0.45, -1.70, 0.0, 0.0] # [0.5] * 7, [1.10, 0.59, 0.06, 1.0, -1.70, 0.0, 0.0]
    trajektorie.points[4].velocities = [0.0] * 7
    trajektorie.points[4].time_from_start = rospy.Duration(10,0)
    trajektorie.points[5].positions = [1.1, 0.59, 0.06, 1.0, -1.70, 0.0, 0.0] # [0.5] * 7, [1.10, 0.59, 0.06, 1.0, -1.70, 0.0, 0.0]
    trajektorie.points[5].velocities = [0.0] * 7
    trajektorie.points[5].time_from_start = rospy.Duration(12,0)
    # definovani id cile
    goal_id = GoalID()
    goal_id.id = "Cil cislo 1"

    # definovani struktury cile pro clienta
    cil = FollowJointTrajectoryGoal()
    cil.trajectory = trajektorie
    #cil.path_tolerance = [0.01] * 7
    #cil.goal_tolerance = [0.01] * 7
    cil.goal_time_tolerance = rospy.Duration(1,0)
    
    # odesli definovany cil k vykonani
    # pub.publish(cil)
    rospy.loginfo(cil)
    arm_action_client.send_goal(cil, feedback_cb=feedback_cb)
    arm_action_client.wait_for_result(rospy.Duration(30,0))

    result = arm_action_client.get_result()
    return result

if __name__ == '__main__':

    try:
        """
        rospy.init_node("action_client")
        wait_for_valid_time(10.0)
        result = hand_point_gesture()
        print("Sevreni ruky: %s" % result)
        result = move_left_arm()
        print("Vysledek pohybu je: %s" % result)
        """
        #hand_left_action_client = create_action_client('/hand_left_controller/follow_joint_trajectory')
        group = init_cartesian_control_for_group("arm_left") #_torso
        #hand_point_gesture()
        moveit_cartesinan_planner(group, 0.6, 0.3, 0.8, 0, 0, 0) # base position
        #moveit_cartesinan_planner(group, 0.2, 0.6471, 1.2765, 0, 40.97, 45) # first picture
        #moveit_cartesinan_planner(group, 0.2, 0.4927, 0.4169, 0, -31.27, 45) # second picture
        moveit_cartesinan_planner(group, 0.4, 0.4476, 0.8342, 0, 86.63, 0) # new random picture

        
    except rospy.ROSInterruptException as e:
        print("Neco se pokazilo: %s" % e)