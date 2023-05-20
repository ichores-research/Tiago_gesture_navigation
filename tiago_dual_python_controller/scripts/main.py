#!/usr/bin/env python

"""
Main script containing functions for controlling individual parts of TIAGo++ robot, communication channels
for sharing image data and info about human, communication with robot via ros topics
and also contains a few demo programs accesed by optional launch argument --demo

List of demo programs: 
(usage: --demo <demo_name>)
- move_head: Program moves only the head of a robot, turns it to several positions and then back to the initial position
- rotate_base: Robot rotates on place to 45 degrees to both directions and then to initial direction
- move_forward_and_back: Robot moves 1 meter forwars, turns around, moves one meter back and turns back. 
- watch_human - Robot tries to find human in its workspace, then tries to keep him in his field of view
"""
import rospy
import cv2
import threading
import base64
import sys
import select
import os
import numpy as np
import argparse

from math import radians, degrees, atan2, sqrt, pow, pi, sin, cos

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose, Quaternion, Point, Twist, TransformStamped
from std_msgs.msg import String
from sensor_msgs.msg import Image, JointState, CameraInfo
from control_msgs.msg import FollowJointTrajectoryAction, JointTolerance
from control_msgs.msg._FollowJointTrajectoryGoal import FollowJointTrajectoryGoal
from nav_msgs.msg import Odometry

from actionlib.simple_action_client import SimpleActionClient

import tf
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from tf.listener import TransformListener

from cv_bridge import CvBridge, CvBridgeError

from image_share_service.service_client import ServiceClient
from classes import LocationPoint, HumanInfo, Location3DJoint

bridge = CvBridge()
look_at_image = False

# Dictionary of angle limit on certain robot joints, not used
ROBOT_JOINTS_MAX_VALUES_RADIANS = {
    "head_1_joint": (-1.24, 1.24),
    "head_2_joint": (-0.98, 0.72),
}

OUT_PATH = "/home/guest/image_recog_results/"

# Definitions of speed of the movement of the robot base
LINEAR_MOVEMENT_SPEED = 0.5
ANGULAR_MOVEMENT_SPEED = 0.5
POSITION_AND_ORIENTATION_TOLERANCE = 0.1
SPEED_TOLERANCE = 0.01
HEAD_JOINT_ANGLE_LIMIT = 70

class Camera:
    """
    Class with information on used camera in the robot
    """
    def __init__(self):
        """
        Info is obtained from topic provided by the robot
        """
        self.camera_info_msg = rospy.wait_for_message("/xtion/rgb/camera_info", CameraInfo)
        self.fx_ = self.camera_info_msg.K[0]
        self.fy_ = self.camera_info_msg.K[4]
        self.cx_ = self.camera_info_msg.K[2]
        self.cy_ = self.camera_info_msg.K[5]
        self.w_ = self.camera_info_msg.width
        self.h_ = self.camera_info_msg.height
        tf_listener.waitForTransform("/xtion_rgb_optical_frame", "/base_footprint", time=rospy.Time(0), timeout=rospy.Duration(5,0))

    def find_distance_to_human(self, human_center_location):
        """
        Function that finds distance from robot base to human from depth image
        Result: distance parameter written to human_info object.
        """
        rospy.loginfo("Finding distance to human in the picture...")
        if human_center_location.x == None or human_center_location.y == None :
            rospy.logerr("No human in the picture!")
            return
        
        # Pixel coordinates of human in the picture.
        pixel_x = int(round(human_center_location.x * self.w_))
        pixel_y = int(round(human_center_location.y * self.h_))
        
        # Reading distance data on position of the center of human in depth image.
        depth_image_message = rospy.wait_for_message("/xtion/depth_registered/image_raw", Image)
        depth_image = image_converter.bridge.imgmsg_to_cv2(depth_image_message, desired_encoding="passthrough")
        distance = depth_image[pixel_y][pixel_x]

        # Camera coordinate frame offset in 2D from base_link
        try:
            [trans, rot] = tf_listener.lookupTransform("/xtion_rgb_optical_frame", "/base_footprint", rospy.Time(0))
        except tf.LookupException or tf.ConnectivityException or tf.ExtrapolationException as e:
            rospy.logerr("Error while looking for transformation: %s" % e)
            return
        
        camera_offset = sqrt(pow(trans[0], 2) + pow(trans[2], 2))
        distance += camera_offset
        rospy.loginfo("Distance of human from the robot: %f" % distance)
        human_info.distance_from_robot = distance

    def get_camera_info(self):
        """
        Function that prints the CameraInfo message with all camera parameters 
        """
        rospy.loginfo("Camera information:\n%s" % self.camera_info_msg)


class ImageConverter:
    """
    Class for converting images from ROS topics and its processing
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.bridge = CvBridge()
        self.look_at_image = False
        self.last_saved_image = None
        self.last_image_buffer = None
        self.camera_resolution = None

        self.image_thread = threading.Thread(target=self.image_listener)
        self.image_thread.start()

    def image_listener(self):
        rospy.Subscriber("/xtion/rgb/image_raw", Image, self.callback_image)
        rospy.spin()

    def callback_image(self,data):
        """
        Callback function called when new image was received
        """
        if not self.get_look_at_image():
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data)
        except CvBridgeError as e:
            rospy.loginfo(e)

        # Transforming image to string for sending it to other running programs
        encoded, image_buffer = cv2.imencode(".jpg", cv_image)
        image_as_txt = base64.b64encode(image_buffer)
        self.lock.acquire()
        self.last_saved_image = cv_image
        self.last_image_as_txt = image_as_txt
        self.look_at_image = False
        self.lock.release()

    def get_last_saved_image(self):
        self.lock.acquire()
        last_image = self.last_saved_image
        self.lock.release()
        return last_image

    def get_last_image_as_txt(self):
        self.lock.acquire()
        image_as_txt = self.last_image_as_txt
        self.lock.release()
        return image_as_txt
    
    def get_look_at_image(self):
        self.lock.acquire()
        look_at_image = self.look_at_image
        self.lock.release()
        return look_at_image
    
    def set_look_at_image(self, value):
        """
        Sets variable look_at_image to bool value -> we save next received image.
        """
        self.lock.acquire()
        self.look_at_image = value
        self.lock.release()

    def get_current_view(self, show=False, save=False, filename=None):
        """
        Function for visualising or saving current image from robot camera.
        Input:
            - show: if True, shows image window with latest image
            - save: if True, saves current image to OUT_PATH with name <filename>
        """
        self.set_look_at_image(True)
        while(self.get_look_at_image()):
            rospy.sleep(0.001)
        self.lock.acquire()
        cv_image = self.last_saved_image
        self.lock.release()

        if show:
            cv2.imshow("Image window", cv_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if not save:
            return
        if filename == None:
            filename = "image_main.jpg"
        annotated_image = cv_image.copy()
        cv2.imwrite(OUT_PATH + filename, annotated_image)


class LatestJointStates:
    """
    Class with listener to last joints positions topic
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.name = []
        self.position = []
        self.velocity = []
        self.effort = []
        self.thread = threading.Thread(target=self.joint_states_listener)
        self.thread.start()
        
    #thread function: listen for joint_states messages
    def joint_states_listener(self):
        rospy.Subscriber('joint_states', JointState, self.joint_states_callback)
        rospy.spin()

    #callback function: when a joint_states message arrives, save the values
    def joint_states_callback(self, msg):
        self.lock.acquire()
        self.name = msg.name
        self.position = msg.position
        self.velocity = msg.velocity
        self.effort = msg.effort
        self.lock.release()

    #returns (found, position, velocity, effort) for the joint joint_name 
    #(found is 1 if found, 0 otherwise)
    def return_joint_state(self, joint_name):

        #no messages yet
        if self.name == []:
            rospy.logerr("No robot_state messages received!\n")
            return (0, 0., 0., 0.)

        #return info for this joint
        self.lock.acquire()
        if joint_name in self.name:
            index = self.name.index(joint_name)
            position = self.position[index]
            velocity = self.velocity[index]
            effort = self.effort[index]

        #unless it's not found
        else:
            rospy.logerr("Joint %s not found!", (joint_name,))
            self.lock.release()
            return (0, 0., 0., 0.)
        self.lock.release()
        return (1, position, velocity, effort)
    
    def return_head_joints_positions(self):
        """
        Returns only positions of both joints in robot head
        """
        head_1_joint_pos = self.return_joint_state("head_1_joint")[1]
        head_2_joint_pos = self.return_joint_state("head_2_joint")[1]

        return [head_1_joint_pos, head_2_joint_pos]


class BasePositionWithOrientation():
    """
    Class representing current position and orientation of robot base
    data are actualised and are expressed in odom coordinate system
    - position_x: base position on the x-axis
    - position_y: base position on the y-axis
    - rotation: angle of current rotation in radians, between -pi to pi
    """
    def __init__(self):
        """
        Setting a thread to listen to odometry data
        """
        self.lock = threading.Lock()
        self.position_x = None
        self.position_y = None
        self.rotation = None
        self.speed_linear = None
        self.speed_angular = None
        self.pub_velocity_command = rospy.Publisher("/mobile_base_controller/cmd_vel", Twist, queue_size=1)
        self.thread = threading.Thread(target=self.odometry_listener)
        self.thread.start()

    def callback_compute_new_odometry(self, msg):
        """
        Callback function computein new odometry data from message in /mobile_base_controller/odom topic
        """
        self.lock.acquire()
        self.position_x = msg.pose.pose.position.x
        self.position_y = msg.pose.pose.position.y
        self.speed_linear = msg.twist.twist.linear.x
        self.speed_angular = msg.twist.twist.angular.z

        rot_quat = msg.pose.pose.orientation
        (roll, pitch, self.rotation) = euler_from_quaternion([rot_quat.x, rot_quat.y, rot_quat.z, rot_quat.w])
        self.lock.release()

    def odometry_listener(self):
        rospy.Subscriber("/mobile_base_controller/odom", Odometry, self.callback_compute_new_odometry)
        rospy.spin

    def final_position_and_angle(self):
        rospy.loginfo("Movement completed. Final position of robot: \nposition_x: %s\nposition_y: %s\nrotation: %s" 
                        % (self.position_x, self.position_y, self.rotation))
        
    def current_position_and_angle(self):
        rospy.loginfo("Current position of robot:\nposition_x: %s\nposition_y = %s\nrotation = %s"
                        % (self.position_x, self.position_y, self.rotation))
        
    def current_speed(self):
        rospy.loginfo("Current speed of robot:\nlinear: %s\nangular = %s"
                        % (self.speed_linear, self.speed_angular))
        
    def get_current_speed(self):
        """
        Function that returns current linear and angular speed of the robot in array
        """
        self.lock.acquire()
        base_speed = [self.speed_linear, self.speed_angular]
        self.lock.release()
        return base_speed

    def get_current_position_and_rotation(self):
        """
        Function that returns current position of the robot in array
        """
        self.lock.acquire()
        position_and_orientation = [self.position_x, self.position_y, self.rotation]
        self.lock.release()
        return position_and_orientation


def parse_args():
    """
    Function for parsing launch arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', type=str, default="", help='launches one of several Tiago++ motion demos')
    parser.add_argument("--debug", type=bool, default=False, help='enables saving of images from tiago camera')
    opts = parser.parse_args()
    return opts


def feedback_cb(msg):
    """
    Basic callback function for printing incoming msg. Used in action clients callback
    - msg: message received
    """
    rospy.loginfo("Received message: \n%s" % msg)
    rospy.sleep(0.2)


def create_action_client(action_topic_interface_string):
    """
    Function that creates action client based on action_topic_interface_string for controlling parts of a robot
    - action_topic_interface_string: string with a name of the topic, for which the client should be created
    Returns: SimpleActionClient() object for certain topic
    """
    action_client = SimpleActionClient(action_topic_interface_string, FollowJointTrajectoryAction)
    iterations = 0
    max_iterations = 3
    rospy.loginfo("Creating action client to topic: %s ..." % action_topic_interface_string)
    while(not action_client.wait_for_server(rospy.Duration(5,0)) and iterations < max_iterations):
        rospy.loginfo("Creating action client to topic: %s ..." % action_topic_interface_string)
        iterations+= 1
    if ( iterations == max_iterations ):
        rospy.logerr("Action client not created!")
        exit(200)
    else:
        rospy.loginfo("Action client created")

    return action_client


def move_head(head_action_client, head_1_joint_degrees, head_2_joint_degrees, wait_for_result=True):
    """
    Function for controlling the head of the robot, its rotation in both of its joints to given angle.

    There are two joints in the head, input values of desired final angle are inputed in degrees
    Inputs: 
        - head_1_joint: horizontal rotation, measured from looking straight forward, positive direction while rotating left
        - head_2_joint: vertical rotation, measured from looking straight forward, positive direction while rotating up
        - wait_for_result: if False, we dont wait for robot to finish rotating head and we continue to execute program
    Use example:
        move_head(head_action_client, 45, 0, wait_for_result=True)
    """

    # First we set the trajectory, one point in our case for robot to rotate to.
    trajektorie = JointTrajectory()
    trajektorie.header.frame_id = "head_movement"
    trajektorie.header.seq += 1
    trajektorie.header.stamp = rospy.Time().now()
    trajektorie.joint_names = ["head_1_joint","head_2_joint"]
    trajektorie.points = [JointTrajectoryPoint()]
    trajektorie.points[0].positions = [radians(head_1_joint_degrees), radians(head_2_joint_degrees)]
    trajektorie.points[0].velocities = [0.0] * 2
    trajektorie.points[0].time_from_start = rospy.Duration(1,0)

    # We set the created trajectory as our goal
    cil = FollowJointTrajectoryGoal()
    cil.trajectory = trajektorie
    cil.goal_time_tolerance = rospy.Duration(0,0)

    # sending goal to created action client
    rospy.loginfo(cil)
    rospy.loginfo("Rotated head to required joint angles:\nhead_1_joint:%f\nhead_2_joint: %f" % (head_1_joint_degrees, head_2_joint_degrees))
    head_action_client.send_goal(cil, feedback_cb=feedback_cb)

    # if we want to wait for the end of head rotation, we wait
    if wait_for_result:
        head_action_client.wait_for_result(rospy.Duration(10,0))
        result = head_action_client.get_result()
        if result:
            rospy.loginfo("Head movement successfully completed!")
        else:
            rospy.logerr("Error while rotating head joints!")


def move_base(goal, final_angle_in_degs, on_place=False):
    """
    Function for moving mobile base around the workspace via velocity commands.
    Inputs: - goal: Point() object with (x,y) coordinates of required final point in odom coordinate system.
            - final_angle_in_degs: final angle (in degrees) for base to rotate to once it reaches the goal.
            - on_place: If True, robot only rotates base to final_angle_in_degs without moving to specified Goal
    - Usage example:
        (for moving to specified goal)
        move_base(Point(x=1, y=1), 120)
        (for rotating in place)
        move_base(Point(), 60, on_place=True)
    """
    # Rate in which we repeat sendig command to base to move, required for smooth movement
    r = rospy.Rate(4)
    speed = Twist()

    # Only angles in range [-pi,pi] can be reached.
    if final_angle_in_degs > 180:
        final_angle_in_degs -= 360
    elif final_angle_in_degs < -180:
        final_angle_in_degs += 360
    # First we try to get to desired goal by using a simple controller.
    # If on_place, we skip the movement part of this process and only rotate to desired angle.
    if on_place:
        rospy.loginfo("Started a rotation of robot on place to desired rotation: \n%s" % (final_angle_in_degs))
    else:
        rospy.loginfo("Started movement of robot to goal position with final rotation: \nposition_x: %s\nposition_y = %s\nrotation = %s" % 
                      (goal.x, goal.y, final_angle_in_degs))
    while not rospy.is_shutdown() and not on_place:

        inc_x = goal.x - mobile_base.position_x
        inc_y = goal.y - mobile_base.position_y

        # If we are close enough to our current goal, we start the rotation to final angle
        if sqrt(pow(inc_x, 2) + pow(inc_y, 2)) < POSITION_AND_ORIENTATION_TOLERANCE:
            speed.linear.x = 0.0
            speed.angular.z = 0.0
            mobile_base.pub_velocity_command.publish(speed)
            break

        angle_to_goal = atan2(inc_y, inc_x)

        # If the robot does not face the required direction, rotate to match it
        if ((angle_to_goal - mobile_base.rotation > POSITION_AND_ORIENTATION_TOLERANCE 
                and abs(angle_to_goal - 2*pi - mobile_base.rotation) > abs(angle_to_goal - mobile_base.rotation))
                or (abs(angle_to_goal + 2*pi - mobile_base.rotation) < abs(angle_to_goal - mobile_base.rotation) 
                and abs(angle_to_goal - mobile_base.rotation) > POSITION_AND_ORIENTATION_TOLERANCE)):
            speed.linear.x = 0.0
            speed.angular.z = ANGULAR_MOVEMENT_SPEED
        elif (angle_to_goal - mobile_base.rotation < -POSITION_AND_ORIENTATION_TOLERANCE 
                or (abs(angle_to_goal - 2*pi - mobile_base.rotation) < abs(angle_to_goal - mobile_base.rotation)
                and abs(angle_to_goal - mobile_base.rotation) > POSITION_AND_ORIENTATION_TOLERANCE)):
            speed.linear.x = 0.0
            speed.angular.z = -ANGULAR_MOVEMENT_SPEED
        # If it does, simply drive towards it
        else:
            speed.linear.x = LINEAR_MOVEMENT_SPEED
            speed.angular.z = 0.0

        mobile_base.pub_velocity_command.publish(speed)
        mobile_base.current_position_and_angle()
        r.sleep()

    # When final goal is reached, start a rotation to final angle
    if not on_place:
        rospy.loginfo("Robot has reached desired goal. Waiting for robot to stop the movement...")
    final_rotation = radians(final_angle_in_degs)

    # We wait for robot to stop the inertial movement when goal position reached
    while(abs(mobile_base.get_current_speed()[0]) > SPEED_TOLERANCE 
            or abs(mobile_base.get_current_speed()[1]) > SPEED_TOLERANCE):
        speed.angular.z = 0.0
        mobile_base.pub_velocity_command.publish(speed)
        rospy.sleep(0.01)

    while not rospy.is_shutdown():
        # While te robot is not facing the correct way, rotate to it.
        if ((final_rotation - mobile_base.rotation > POSITION_AND_ORIENTATION_TOLERANCE 
                and abs(final_rotation - 2*pi - mobile_base.rotation) > abs(final_rotation - mobile_base.rotation))
                or (abs(final_rotation + 2*pi - mobile_base.rotation) < abs(final_rotation - mobile_base.rotation)
                and abs(final_rotation - mobile_base.rotation) > POSITION_AND_ORIENTATION_TOLERANCE)):
            speed.angular.z = ANGULAR_MOVEMENT_SPEED
        elif (final_rotation - mobile_base.rotation < -POSITION_AND_ORIENTATION_TOLERANCE 
                or (abs(final_rotation - 2*pi - mobile_base.rotation) < abs(final_rotation - mobile_base.rotation)
                and abs(final_rotation - mobile_base.rotation) > POSITION_AND_ORIENTATION_TOLERANCE)):
            speed.angular.z = -ANGULAR_MOVEMENT_SPEED
        # When final angle reached, stop the rotation and show the final position of the robot
        else:
            speed.angular.z = 0.0
            break
        mobile_base.pub_velocity_command.publish(speed)
        mobile_base.current_position_and_angle()
        mobile_base.current_speed()
        r.sleep()
    rospy.loginfo("Robot is facing final direction. Waiting for robot to stop the rotation...")
    while(abs(mobile_base.get_current_speed()[0]) > ANGULAR_MOVEMENT_SPEED
            or abs(mobile_base.get_current_speed()[1]) > ANGULAR_MOVEMENT_SPEED):
        speed.angular.z = 0.0
        mobile_base.pub_velocity_command.publish(speed)
        rospy.sleep(0.01)

    mobile_base.final_position_and_angle()


def analyze_picture_with_mediapipe(step=None):
    """
    Function that sends current image from camera to process it by MediaPipe and writes results to HumanInfo() object.
    """
    image_converter.set_look_at_image(True)
    if step:
        rospy.loginfo("Analysing camera image by MediaPipe in step %d ..." % int(step))
    else:
        rospy.loginfo("Analysing camera image by MediaPipe...")
    # While the image is being processed, we wait
    while(image_converter.get_look_at_image()):
        rospy.sleep(0.001)
    
    # Converting image data to string for transfer to another process through ServiceClient() object
    cv_image_as_txt = image_converter.get_last_image_as_txt()
    data = {
        "Hello": "MediaPipe",
        "image": cv_image_as_txt
    }
    results = mediapipe_image_service_client.call(data)

    # Saving received results to HumanInfo() object, if any human was detected
    try:
        human_info.human_found = results["human_found"]
        human_info.human_centered = results["human_centered"]
        human_info.final_robot_rotation.x = results["final_robot_rotation.x"]
        human_info.final_robot_rotation.y = results["final_robot_rotation.y"]
        human_info.center_location.x = results["human_center_x"]
        human_info.center_location.y = results["human_center_y"]
    except KeyError:
        rospy.logerr("Returned dict from MediaPipe server is empty!")


def analyze_picture_with_alphapose():
    """
    Function that sends current image from robot to AlphaPose image server
    """
    rospy.loginfo("Analysing camera image by AlphaPose...")
    image_converter.set_look_at_image(True)
    while(image_converter.get_look_at_image()):
        rospy.sleep(0.001)
    cv_image_as_txt = image_converter.get_last_image_as_txt()
    data = {
        "Hello": "AlphaPose",
        "image_name": "human_alphapose.jpg",
        "image_as_txt": cv_image_as_txt,
        "out_path": OUT_PATH,
    }
    # Sending data to AlphaPose image server, waits for response and saves the result
    response = alphapose_image_service_client.call(data)
    human_info.alphapose_json_result = response["json_results"]
    rospy.loginfo("Analysis of pose via AlphaPose finished. Results:")
    print(human_info.alphapose_json_result)


def analyze_picture_with_motionbert():
    """
    Function that sends received json data from AlphaPose to MotionBERT image server
    """
    rospy.loginfo("Zahajeni analyzy obrazu pomoci MotionBERT...")
    data = {
        "Hello": "MotionBERT",
        "out_path": OUT_PATH,
        "json_results": human_info.alphapose_json_result,
    }
    # Sending data to MotionBERT image server, waits for response and saves the result
    response = motionbert_image_service_client.call(data)
    human_info.motionbert_joints_positions = response["human_joints_positions"]
    rospy.loginfo("Analysis of pose via MotionBERT finished. Results:")
    rospy.loginfo(human_info.motionbert_joints_positions)
    

def find_human():
    """
    Function that performs a sequence of moves, which schould lead to finding human in workspace
    Sequence description:
        - step 0: Robot rotates the base to match the head rotation.
        - step 1: Robot rotates the head to 45 degrees
        - step 2: Robot rotates the head to -45 degrees
        - step 3: Robot spins on place to -135 degrees from the first direction
        - step 4: Robot rotates the head to 0 degrees
        - step 5: Robot rotates the head to 45 degrees
        - step 6: Robot spins on place to 90 degrees from the first direction
        - step 7: Robot rotates the head to 0 degrees
        - step 8: Robot spins on place to the first direction
        - Afrer each step, 3 pictures are taken and processed by MediaPipe to find human on them.
        - Repeats until a human is found
    """
    tries = 0
    step = 0
    base_rotation = mobile_base.rotation
    
    # Repeat the sequence until we find a human
    while(not human_info.is_found()):

        if step == 0 and (abs(latest_joint_states.return_head_joints_positions()[0]) > 0.01 or abs(latest_joint_states.return_head_joints_positions()[0]) > 0.01):
            head_position = latest_joint_states.return_head_joints_positions()
            base_rotation = base_rotation + head_position[0]
            move_head(head_action_client, 0, 0, wait_for_result=False)
            move_base(Point(), degrees(base_rotation), on_place=True)
        elif step == 1:
            move_head(head_action_client, 45, 0)
        elif step == 2:
            move_head(head_action_client, -45, 0)
        elif step == 3:
            move_base(Point(), degrees(base_rotation)-135, on_place=True)
        elif step == 4:
            move_head(head_action_client, 0, 0)
        elif step == 5:
            move_head(head_action_client, 45, 0)
        elif step == 6:
            move_base(Point(), degrees(base_rotation)+90, on_place=True)
        elif step == 7:
            move_head(head_action_client, 0, 0)
        elif step == 8:
            move_base(Point(), degrees(base_rotation), on_place=True)
            step = 0
            rospy.loginfo("End of search cycle, human not found. Starting new cycle...")
            continue
        
        # Process 3 images taken in each direction.
        tries = 0
        while(not human_info.is_found() and tries < 3):
            analyze_picture_with_mediapipe(step)
            tries += 1

        step += 1
    rospy.loginfo("Human found on picture!")


def center_camera_on_human():
    """
    Function that centers camera on human when found in the picture
    Works with data stored in the HumanInfo() object
    """
    rospy.loginfo("Centering on human in the picture...")
    [head_1_joint_pos, head_2_joint_pos] = latest_joint_states.return_head_joints_positions()
    final_head_hor_angle = degrees(head_1_joint_pos) + human_info.final_robot_rotation.x
    final_head_ver_angle = degrees(head_2_joint_pos) + human_info.final_robot_rotation.y

    # if the human is to close to the limit of head joint rotation, rotate the base to human.
    if abs(final_head_hor_angle) < HEAD_JOINT_ANGLE_LIMIT:
        move_head(head_action_client, final_head_hor_angle, final_head_ver_angle)
    elif abs(final_head_hor_angle) >= HEAD_JOINT_ANGLE_LIMIT:
        move_head(head_action_client, 0, final_head_ver_angle, wait_for_result=False)
        move_base(Point(), degrees(mobile_base.rotation) + final_head_hor_angle, on_place=True)

    head_joints_positions = latest_joint_states.return_head_joints_positions()
    rospy.loginfo("Centering complete. Current head joints positions:\nhead_1_joint: %s\nhead_2_joint: %s"
            % (degrees(head_joints_positions[0]), degrees(head_joints_positions[1])))


def get_point_on_ground(right_shoulder, right_wrist):
    """
    Funcion that computes the intersection of line between right shoulder and right wrist with the ground
    in the MotionBERT human coordiante system
    Inputs:
        - right_shoulder: Location3DJoint() object with positional data
        - right_wrist: Location3DJoint() object with positional data
    Output:
        - final_point: Location3DJoint() object with positional data
    """
    vector = Location3DJoint(
        x = right_wrist.x - right_shoulder.x,
        y = right_wrist.y - right_shoulder.y,
        z = right_wrist.z - right_shoulder.z
    )
    rospy.loginfo("Vector between joints:\n(%f, %f, %f)" %(vector.x, vector.y, vector.z))
    final_point = Location3DJoint(
        x = right_shoulder.x - (right_shoulder.z/vector.z)*vector.x,
        y = right_shoulder.y - (right_shoulder.z/vector.z)*vector.y,
        z = 0
    )
    rospy.loginfo("Final point coordinates:\nx = %f\ny = %f\nz = %f)" %(final_point.x, final_point.y, final_point.z))
    return final_point


def get_rotation_matrix(alpha):
    """
    Function that creates rotation matrix for rotation in 2D by angle alpha
    Input:
        - alpha: rotation in rads
    Output:
        np.array() rotational matrix
    """
    return np.array([
        [cos(alpha),    -sin(alpha),    0],
        [sin(alpha),    cos(alpha),     0], 
        [0,             0,              1]
    ])


def get_translation_matrix(x, y):
    """
    Function that creates translation matrix for translation in 2D by x and y 
    Input:
        - x: Translation to the point in x-axis
        - y: Translation to the point in y-axis
    Output:
        np.array() translation matrix
    """
    return np.array([
        [1,    0,    x],
        [0,    1,    y], 
        [0,    0,    1]
    ])


def transform_points_to_odom(final_point):
    """
    Function transforming location of human and the final point to odom coordinate system
    Input: 
        - final_point: Location3DJoint() object in MotionBERT human coordinate system
    Output: 
        - final_point_matrix: np.array() with coordinates of final point in odom coordinate system
        - human_coordinates: np.array() with coordinates of human in odom coordinate system
    """
    # First, we rotate the coordinate system of human from MotionBERT to match base_link coordinate system
    final_point_matrix = np.array([[final_point.x], [final_point.y], [1]])
    rot_matrix = get_rotation_matrix(-pi/2)
    final_point_matrix = np.matmul(rot_matrix, final_point_matrix)

    # Distance from robot base to human extracted from depth image
    distance_to_human = human_info.distance_from_robot
    # Human coordinates in human coordinate system -> [0, 0, 1]
    human_coordinates = np.array([[0], [0], [1]])
    # Transformation from base of the human to base of the robot
    trans_matrix = get_translation_matrix(distance_to_human, 0)
    final_point_matrix = np.matmul(trans_matrix, final_point_matrix)
    human_coordinates = np.matmul(trans_matrix, human_coordinates)

    # Rotation to match the orientation of odom coordinate system.
    base_position_and_rotation = mobile_base.get_current_position_and_rotation()
    head_joint_position = latest_joint_states.return_head_joints_positions()
    trans_matrix = get_translation_matrix(base_position_and_rotation[0], base_position_and_rotation[1])
    rot_matrix = get_rotation_matrix(base_position_and_rotation[2] + head_joint_position[0])
    final_point_matrix = np.matmul(rot_matrix, final_point_matrix)
    human_coordinates = np.matmul(rot_matrix, human_coordinates)

    # Translation to odom coordinate system.
    final_point_matrix = np.matmul(trans_matrix, final_point_matrix)
    human_coordinates = np.matmul(trans_matrix, human_coordinates)
    rospy.loginfo("Final point after transformation to odom coordinate system:")
    print(final_point_matrix)
    rospy.loginfo("Human position after transformation to odom coordinate system:")
    print(human_coordinates)
    return [final_point_matrix, human_coordinates]


def find_final_point():
    """
    Function processesing human skeleton data from MotionBERT and getting a final point location.
    Output:
        - final_point_matrix: np.array() with positonal data of the desired point (in meters)
        - final_angle_in_degs: float representig final robot rotation of base
    """
    rospy.loginfo("Creating dictionary of found human joint positions...")
    human_info.create_joints_dict()

    rospy.loginfo("Moving oroign of human coordinate system...")
    human_info.change_reference_of_joints()

    rospy.loginfo("Computing intersection of the indicated direction of hand with ground...")
    right_shoulder = human_info.motionbert_joints_positions["RShoulder"]
    right_wrist = human_info.motionbert_joints_positions["RWrist"]
    final_point = get_point_on_ground(right_shoulder, right_wrist)

    rospy.loginfo("Transforming found intersection to odom coordinate system...")
    [final_point_matrix, human_coordinates] = transform_points_to_odom(final_point)
    
    rospy.loginfo("Computing final degree after moveing to final point...")
    inc_x = human_coordinates[0] - final_point_matrix[0]
    inc_y = human_coordinates[1] - final_point_matrix[1]

    final_angle_in_degs = degrees(atan2(inc_y, inc_x))
    rospy.loginfo("Final angle for base to rotate to:\n%f" % final_angle_in_degs)

    # Promt that waits to confirm movement to final point by pressing ENTER
    raw_input("Press ENTER to confirm movement to final point")

    return [final_point_matrix, final_angle_in_degs]


def move_robot_to_final_point(final_point_matrix, final_angle_in_degs):
    """
    Function performing movement of robot to found final point to which human pointed.
    Inputs:
        - final_point_matrix: np.array() with positonal data of the desired point (in meters)
        - final_angle_in_degs: float representig final robot rotation of base
    Result: Robot reaches desired point
    """
    rospy.loginfo("Moving robot to final point position:\nx = %f\ny = %f...")
    move_head(head_action_client, 0, 0, wait_for_result=False)
    move_base(Point(x = final_point_matrix[0], y=final_point_matrix[1]), final_angle_in_degs)

if __name__ == '__main__':

    try:
        rospy.init_node("tiago_controller", anonymous=True)
        
        head_action_client = create_action_client("/head_controller/follow_joint_trajectory")
        
        tf_listener = TransformListener()
        mobile_base = BasePositionWithOrientation()
        image_converter = ImageConverter()
        latest_joint_states = LatestJointStates()
        human_info = HumanInfo()
        camera_info = Camera()

        # Creating clients to different scripts (image processing)
        mediapipe_image_service_client = ServiceClient()
        alphapose_image_service_client = ServiceClient(port=242425)
        motionbert_image_service_client = ServiceClient(port=242426)

        args = parse_args()

        # Demo sequence "camera"
        if args.demo == "camera":
            analyze_picture_with_mediapipe()
            image_converter.get_current_view(show=True)

        # Demo sequence "move_head"
        elif args.demo == "move_head":
            move_head(head_action_client, 0, 0)
            move_head(head_action_client, 30, 0)
            move_head(head_action_client, -30, 0)
            move_head(head_action_client, 0, 0)
            move_head(head_action_client, 0, 30)
            move_head(head_action_client, 0, -30)
            move_head(head_action_client, 0, 0)

        # Demo sequence "rotate_base"
        elif args.demo == "rotate_base":
            current_base_position = mobile_base.get_current_position_and_rotation()
            while(current_base_position[2] == None):
                rospy.sleep(0.1)
                current_base_position = mobile_base.get_current_position_and_rotation()

            move_base(Point(), degrees(current_base_position[2])+45, on_place=True)
            move_base(Point(), degrees(current_base_position[2])-45, on_place=True)
            move_base(Point(), degrees(current_base_position[2]), on_place=True)

        # Demo sequence "move_forward_and_back"
        elif args.demo == "move_forward_and_back":
            current_base_position = mobile_base.get_current_position_and_rotation()
            while(current_base_position[2] == None):
                rospy.sleep(0.1)
                current_base_position = mobile_base.get_current_position_and_rotation()
            rot_matrix = get_rotation_matrix(current_base_position[2])
            trans_matrix = get_translation_matrix(current_base_position[0], current_base_position[1])
            final_point_matrix = np.array([[1], [0], [1]])
            final_point_matrix = np.matmul(rot_matrix, final_point_matrix)
            final_point_matrix = np.matmul(trans_matrix, final_point_matrix)
            move_base(Point(x=final_point_matrix[0], y=final_point_matrix[1]), degrees(current_base_position[2]))
            final_point_matrix = np.array([[0], [0], [1]])
            final_point_matrix = np.matmul(rot_matrix, final_point_matrix)
            final_point_matrix = np.matmul(trans_matrix, final_point_matrix)
            move_base(Point(x=final_point_matrix[0], y=final_point_matrix[1]), degrees(current_base_position[2]))


        else:
            tries = 0
            while not rospy.is_shutdown() or not KeyboardInterrupt:
                if not human_info.is_found() and tries >= 3:
                    tries = 0
                    find_human()
                elif human_info.is_found() and not human_info.is_centered():
                    tries = 0
                    center_camera_on_human()
                elif not human_info.is_found():
                    tries += 1
                elif human_info.is_found() and human_info.is_centered() and args.demo == "":
                    tries = 0
                    rospy.loginfo("Human is in center of the camera view. Press ENTER to regognise pose and move robot...")
                    i, o, e = select.select( [sys.stdin], [], [], 3)
                    if (i):
                        sys.stdin.readline().strip() # Removes ENTER char to wait for it again next round
                        if args.debug:
                            image_converter.get_current_view(show=False, save=True, filename="human.jpg")
                        analyze_picture_with_alphapose()
                        analyze_picture_with_motionbert()
                        mobile_base.current_position_and_angle()
                        camera_info.find_distance_to_human(human_info.center_location)
                        [final_point_matrix, final_angle_in_degs] = find_final_point()
                        move_robot_to_final_point(final_point_matrix, final_angle_in_degs)
                    else:
                        rospy.loginfo("Recentering on human...")
                
                # In demo "watch_human", robot only centers camera on human
                elif human_info.is_found() and human_info.is_centered() and args.demo == "watch_human":
                    tries = 0
                    rospy.loginfo("Human is in center of the camera view.")
                    rospy.sleep(0.5)
                    rospy.loginfo("Recentering camera on human...")
                analyze_picture_with_mediapipe()

        rospy.spin()
        
    except rospy.ROSInterruptException as e:
        rospy.logerr("Something happened: %s" % e)