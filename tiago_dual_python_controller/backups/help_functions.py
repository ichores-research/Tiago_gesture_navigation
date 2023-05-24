#!/usr/bin/env python3

# zaloha s funkcnim publisherem a zpracovanim vsech obrazku naraz
import sys
import rospy
import cv2
import mediapipe.python as mp
import numpy as np
import math
from std_msgs.msg import String

import moveit_msgs.msg
import moveit_commander

ROBOT_HEIGHT = 1.126886 # hodnota vysky hlavy robota v zakladni poloze ze simulace

class LocationPoint:
  def __init__(self, x, y):
    self.x = x
    self.y = y

def change_reference_of_joints(self):
        """
        Function that moves origin of coordinates of human joints to place on the ground between the hips
        """
        center = Location3DJoint()

        left_hip = self.motionbert_joints_positions["LHip"]
        right_hip = self.motionbert_joints_positions["RHip"]
        left_ankle = self.motionbert_joints_positions["LAnkle"]
        right_ankle = self.motionbert_joints_positions["RAnkle"]
        
        center.x = (left_hip.x + right_hip.x)/2
        center.y = (left_hip.y + right_hip.y)/2
        center.z = min(left_ankle.z, right_ankle.z)
        print("Center location: (%f, %f, %f)" %(center.x, center.y, center.z))
        for joint in self.motionbert_joints_positions:
            self.motionbert_joints_positions[joint].x -= center.x
            self.motionbert_joints_positions[joint].y -= center.y
            self.motionbert_joints_positions[joint].z -= center.z
            self.motionbert_joints_positions[joint].z = -self.motionbert_joints_positions[joint].z # chci aby nahoru bylo do kladnych hodnot -> pravotocivy souradny system.
        self.clean_print_joints()

def init_mediapipe():
    mp_drawing = mp.solutions.drawing_utils
    #mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    return mp_drawing, mp_pose # mp_drawing_styles, mp_pose

def recognise_person_from_images(mp_drawing, mp_pose): #, mp_drawing_styles
    """
    Funkce, ktera rozpozna pozu cloveka a navrati souradnice jeho bodu v obdrzenem obraze
    Souradnice v obraze jsou pocitany od leveho horniho rohu tak, ze sirka obrazu je souradnice x, vyska obrazu je souradnice y
    """
    # For static images:
    IMAGE_FILES = ["test_image_man_1.jpg", "test_image_woman_1.jpg", "image_1.jpg", "image_2.jpg"]
    with mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5) as pose:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread("/home/guest/tiago_dual_public_ws/src/python_control_testing_functions/images/" + file) 
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # vypis si souradnice zajimavych bodu
            if not results.pose_landmarks:
                continue
            print("Picture %d: %s" % (idx, file))
            print( 'Nose coordinates: (' +
                str(float(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width)) + " , " +
                str(float(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height)) + " )"
            )
            print( 'Left knee coordinates: (' +
                str(float(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * image_width)) + " , " +
                str(float(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * image_height)) + " )"
            )
            print( 'Right knee coordinates: (' +
                str(float(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * image_width)) + " , " +
                str(float(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * image_height)) + " )"
            )
            print( 'Left hip coordinates: (' +
                str(float(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image_width)) + " , " +
                str(float(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image_height)) + " )"
            )
            print( 'Right hip coordinates: (' +
                str(float(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * image_width)) + " , " +
                str(float(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * image_height)) + " )"
            )
            print( 'Right index finger coordinates: (' +
                str(float(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x * image_width)) + " , " +
                str(float(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y * image_height)) + " )"
            )
            print( 'Right wrist finger coordinates: (' +
                str(float(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width)) + " , " +
                str(float(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_height)) + " )"
            )

            # definice pomocnych bodu
            approximate_base_location = LocationPoint(0,0)
            right_wrist_from_base = LocationPoint(0,0)
            right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

            # zpracovani nalezene pozice na souradnice a uhly nakloneni pro robota, spolu s pribliznym meritkem podle velikosti osoby z obrazku
            human_height, angle = process_returned_points(results, approximate_base_location, right_wrist, right_wrist_from_base)
            print("Relative human height: %f" % human_height)

            print("Angle of index finger: %f" % angle)
            scale = ROBOT_HEIGHT / human_height
            print("Scale: %f" % scale)

            # vytisteni nalezenych dulezitych bodu
            print( 'Approximate TIAGO++ base location coordinates: (' +
                str(float(approximate_base_location.x)) + " , " +
                str(float(approximate_base_location.y)) + " )"
            )
            wrist_position_and_angle_msg = ",".join([str(-right_wrist_from_base.x * 0.7*scale), str(-right_wrist_from_base.y*scale), str(angle)])
            print("Wrist position message: %s" % wrist_position_and_angle_msg)
            print( 'right wrist location from base coordinates scaled: (' +
                str(float(right_wrist_from_base.x * 0.7*scale)) + " , " +
                str(float(right_wrist_from_base.y * scale)) + " )"
            )

            # Draw pose landmarks on the image.
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS)
            cv2.imwrite('/home/guest/tiago_dual_public_ws/src/python_control_testing_functions/tmp/' + str(idx) + '.jpg', annotated_image)

    return wrist_position_and_angle_msg

def process_returned_points(results, approximate_base_location, right_wrist, right_wrist_from_base):
    """
    Funkce ktera z navracenych souradnic obrazovych bodu udela souradnice a natoceni pro ruku robota
    """
    # prejmenuj si dulezite parametry pro vypocty
    left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    right_index = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]

    # priblizna lokace zakladny robota je mezi koleny a boky cloveka
    approximate_base_location.x = (left_knee.x + right_knee.x + left_hip.x + right_hip.x)/4 
    approximate_base_location.y = (left_knee.y + right_knee.y + left_hip.y + right_hip.y)/4

    # vypoctene souraadnice zapesti cloveka od pomyslne zakladny robota, odpovidajici poloze end efektoru robota 
    right_wrist_from_base.x = right_wrist.x - approximate_base_location.x
    right_wrist_from_base.y = right_wrist.y - approximate_base_location.y

    # vypocet vysky cloveka pro korekci ruzne vysky lidi
    human_height = math.sqrt(math.pow(nose.x - approximate_base_location.x, 2) + math.pow(nose.y - approximate_base_location.y, 2))

    # vypocet naklonu ukazovacku v obraze
    angle = math.degrees(math.atan2(right_wrist.y - right_index.y,right_wrist.x - right_index.x))

    return human_height, angle

def talker(wrist_position_and_angle_msg):
    """
    Talker ktery vysila zpravu do kanalu chatter s daty pro pohyb robota
    """
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        rospy.loginfo(wrist_position_and_angle_msg)
        pub.publish(wrist_position_and_angle_msg + " %s" % rospy.get_time())
        rate.sleep()

def callback_received_human_info(msg):
    """
    Callback, ktery je zavolan pri obdrzeni dat o clovekovi v obraze.
    Format zpravy je nasledujici:
    "human_info.human_found,human_info.human_centered,human_info.final_robot_rotation.x,human_info.final_robot_rotation.y"
    """
    rospy.loginfo(rospy.get_caller_id() + " Obdrzena data o clovekovi v obraze: %s", msg.data)
    params = str(msg.data).split(",")
    human_info.human_found = params[0] == "True"
    human_info.human_centered = params[1] == "True"
    if not human_info.is_centered() and human_info.is_found():
        human_info.final_robot_rotation.x = float(params[2])
        human_info.final_robot_rotation.y = float(params[3])

#rospy.Subscriber("human_info_cmd", String, callback_received_human_info)

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

def hand_point_gesture(hand_action_client):
    """
    Funkce, ktera ruku robota nastavi do tvaru ukazovani pomoci ukazovacku
    - hand_action_client: vytvoreny klient pro ovladani ruky
    """
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
    cil.goal_tolerance = [JointTolerance()]
    cil.goal_tolerance[0].position = 0.01

    rospy.loginfo(cil)
    hand_action_client.send_goal(cil, feedback_cb=feedback_cb)
    hand_action_client.wait_for_result(rospy.Duration(10,0))

    result = hand_action_client.get_result()
    return result

#hand_point_gesture(hand_left_action_client)

def moveit_cartesian_planner(group, coord_x, coord_y, coord_z, rot_x, rot_y, rot_z):
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
    
    #print(orientation)
    orientation_quat = Quaternion()
    orientation_quat.x = orientation[0] 
    orientation_quat.y = orientation[1] 
    orientation_quat.z = orientation[2] 
    orientation_quat.w = orientation[3]
    pose_goal.orientation = orientation_quat
    group.set_pose_target(pose_goal)

    # naplanuj a proved
    plan = group.go(wait=True)
    #print(plan)
    group.stop()
    group.clear_pose_targets()
    return plan

def callback_received_robot_coordinates(data):
    """
    Funkce ktera se zavola pri prijmu zpravy z tiago_left_wrist_position topicu
    """
    rospy.loginfo(rospy.get_caller_id() + " I heard %s", data.data)
    rospy.loginfo("Zacinam pohyb leve ruky robota podle obdrzenych souradnic")
    souradnice = str(data.data)
    souradnice = souradnice.split(",")
    print(souradnice)
    coord_x = float(souradnice[0])
    coord_y = float(souradnice[1])
    coord_z = float(souradnice[2])
    rot_y = float(souradnice[3])
    rot_z = float(souradnice[4])
    result = moveit_cartesian_planner(group, coord_x, coord_y, coord_z, 0, rot_y, rot_z)
    if result == True:
        rospy.loginfo("Pohyb robota uspesne dokoncen")
    else:
        rospy.loginfo("Chyba pri pohybu robota")

# rospy.Subscriber("tiago_left_wrist_position", String, callback_received_robot_coordinates)

def init_cartesian_control_for_group(group_name):
    """
    Funkce innicializujici nezbytne promenne pro moznost ovladani robota a skupiny kloubu pomoci MoveIt
    - group_name: string s nazvem skupiny kloubu, pro ktere se ma vytvorit MoveGroupCommander
    """
    moveit_commander.roscpp_initialize(sys.argv)
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

#group = init_cartesian_control_for_group("arm_left") #_torso

if __name__ == '__main__':

    mp_drawing, mp_pose =  init_mediapipe()
    IMAGE_FILES = ["test_image_man_1.jpg", "test_image_woman_1.jpg", "image_1.jpg", "image_2.jpg"]
    wrist_position_and_angle_msg = recognise_person_from_images(mp_drawing, mp_pose)
    try:
        talker(wrist_position_and_angle_msg)
    except rospy.ROSInterruptException:
        pass