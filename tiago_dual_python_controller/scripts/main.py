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
        if human_center_location.x == None or human_center_location.y == None :
            print("No human in the picture")
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
    Trida zahrnujici konverzi obrazu z ROS topicu a jeho zpracovavani 
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
        Sets variable look_at_image to bool value
        """
        self.lock.acquire()
        self.look_at_image = value
        self.lock.release()

    def get_current_view(self, show=False, save=False, filename=None):
        """
        Function for visualising or saving current image from robot camera.
        Input:  show - if True, shows image window with latest image
                save - if True, saves current image to OUT_PATH with name <filename>
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
    - position_x: x-ova souradnice zakladny v metrech
    - position_y: y-ova souradnice zakladny v metrech
    - rotation: rotace zakladny v radianech
    """
    def __init__(self):
        """
        Nastaveni pocatecni polohy pred zacatkem pohybu.
        """
        self.lock = threading.Lock()
        self.position_x = None
        self.position_y = None
        self.rotation = None
        self.speed_linear = None
        self.speed_angular = None
        self.thread = threading.Thread(target=self.odometry_listener)
        self.thread.start()

    def callback_compute_new_odometry(self, msg):
        """
        Funkce ktera spocita novou odometrii na zaklade dat z topicu /mobile_base_controller/odom
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
        """
        Funkce vypisujici vyslednou polohu zakladny po pohybu robota.
        """
        rospy.loginfo("Pohyb dokoncen, vysledna poloha robota je: \nposition_x: %s\nposition_y: %s\nrotation: %s" 
                        % (self.position_x, self.position_y, self.rotation))
        
    def current_position_and_angle(self):
        """
        Funkce vypisujici aktualni souradnice zakladny robota behem jejiho pohybu
        """
        rospy.loginfo("Aktualni pozice zakladny robota:\nposition_x: %s\nposition_y = %s\nrotation = %s"
                        % (self.position_x, self.position_y, self.rotation))
        
    def current_speed(self):
        rospy.loginfo("Aktualni rychlost pohybu zakladny robota:\nlinear: %s\nangular = %s"
                        % (self.speed_linear, self.speed_angular))
        
    def get_current_speed(self):
        """
        Funkce vracejici rychlosti zakladny robota v poli
        """
        self.lock.acquire()
        base_speed = [self.speed_linear, self.speed_angular]
        self.lock.release()
        return base_speed

    def get_current_position_and_rotation(self):
        """
        Funkce vracejici pozici a orientaci vypoctenou z odometrie.
        """
        self.lock.acquire()
        position_and_orientation = [self.position_x, self.position_y, self.rotation]
        self.lock.release()
        return position_and_orientation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', type=str, default="", help='launches one of several Tiago++ motion demos')
    opts = parser.parse_args()
    return opts


def feedback_cb(msg):
    """
    Obecna callback funkce pro vypis stavu ovladanych kloubu pri jejich ovladani pomoci akcnich klientu
    - msg: zprava navracena akcnim klientem
    """
    print("Zpetna vazba: \n%s" % msg)
    rospy.sleep(0.2)


def create_action_client(action_topic_interface_string):
    """
    Funkce, ktera podle zadaneho stringu vytvori action client pro ovladani urcitych casti robota
    - action_topic_interface_string: string s nazvem topicu, pro ktery se ma vytvorit akcni klient
    """
    action_client = SimpleActionClient(action_topic_interface_string, FollowJointTrajectoryAction)
    iterations = 0
    max_iterations = 3
    rospy.loginfo("Creating action client to topic: %s ..." % action_topic_interface_string)
    while(not action_client.wait_for_server(rospy.Duration(5,0)) and iterations < max_iterations):
        rospy.loginfo("Creating action client to topic: %s ..." % action_topic_interface_string)
        iterations+= 1
    if ( iterations == max_iterations ):
        rospy.loginfo("Action client not created!")
        exit(200)
    else:
        rospy.loginfo("Action client created")

    return action_client


def move_head(head_action_client, head_1_joint_degrees, head_2_joint_degrees, wait_for_result=True):
    """
    Funkce pro ovladani hlavy robota pro natoceni do urciteho smeru

    V hlave se pouzivaji dva klouby, hodnoty jejich natoceni se zadavaji ve *stupnich*
    - head_1_joint: pohyb po horizontu, do stran, kladny smer je doleva, pocatecny hodnota je 0 - primo vpred
    - head_2_joint: pohyb nahoru a dolu, kladny smer nahoru, pocatecni poloha je 0 - vodorovne
    - Pokud je wait_for_result=False, funkce neceka na konec pohybu hlavy a pokracuje ve vykonavani programu
    Priklad pouziti:
    move_head(head_action_client, 45, 0, wait_for_result=False)
    """

    # nejprve si nastavim trajektorii, neboli v nasem pripade jeden bod, do ktereho se ma robot dostat
    trajektorie = JointTrajectory()
    trajektorie.header.frame_id = "Pohyb hlavy"
    trajektorie.header.seq += 1
    trajektorie.header.stamp = rospy.Time().now()# + rospy.Duration(1,0)
    trajektorie.joint_names = ["head_1_joint","head_2_joint"]
    trajektorie.points = [JointTrajectoryPoint()]
    trajektorie.points[0].positions = [radians(head_1_joint_degrees), radians(head_2_joint_degrees)]
    trajektorie.points[0].velocities = [0.0] * 2
    trajektorie.points[0].time_from_start = rospy.Duration(1,0)

    # nastavime ho jako cil
    cil = FollowJointTrajectoryGoal()
    cil.trajectory = trajektorie
    cil.goal_time_tolerance = rospy.Duration(0,0)

    # cil poslu do vytvoreneho akcniho klienta
    rospy.loginfo(cil)
    rospy.loginfo("Zacinam pohyb hlavy do uhlu (%s, %s)" % (str(head_1_joint_degrees), str(head_2_joint_degrees)))
    head_action_client.send_goal(cil, feedback_cb=feedback_cb)

    # pokud chci pockat na konec pohybu, zhodnotim, zda-li se pohyb provedl spravne ci nikoli
    if wait_for_result:
        head_action_client.wait_for_result(rospy.Duration(10,0))
        result = head_action_client.get_result()
        if result:
            rospy.loginfo("Pohyb hlavy uspesne dokoncen.")
        else:
            rospy.logerr("Neco se pokazilo na pohybu hlavy...")


def move_base(goal, final_angle_in_degs, on_place=False):
    """
    Funkce, ktera pomoci velocity command bude pohybovat zakladnou robota
    - goal: Point() udavajici (x,y) souradnici pozadovaneho bodu k dosazeni robotem
    - Priklad pouziti
    move_base(Point(x=0, y=0), 120, on_place=True)
    """
    # Rychlost opakovani vysilani povelu k jizde, nutne pro plynuly pohyb
    r = rospy.Rate(4)
    speed = Twist()

    if final_angle_in_degs > 180:
        final_angle_in_degs -= 360
    elif final_angle_in_degs < -180:
        final_angle_in_degs += 360
    # nejprve se snazim jednoduchym ovladacem dojet na misto ktere je v goal
    # pokud je nastaven argument on_place, robot se ma pouze otocit na miste do urciteho uhlu
    if on_place:
        rospy.loginfo("Zahajuji otoceni robota na miste do uhlu: \nrotation = %s" % (final_angle_in_degs))
    else:
        rospy.loginfo("Zahajuji pohyb robota na souradnice s otocenim do finalniho uhlu: \nposition_x: %s\nposition_y = %s\nrotation = %s" % 
                      (goal.x, goal.y, final_angle_in_degs))
    while not rospy.is_shutdown() and not on_place:

        inc_x = goal.x - mobile_base.position_x
        inc_y = goal.y - mobile_base.position_y

        # pokud uz jsem dostatecne blizko cili, pak se presun na otoceni na finalni misto
        if sqrt(pow(inc_x, 2) + pow(inc_y, 2)) < 0.1:
            speed.linear.x = 0.0
            speed.angular.z = 0.0
            pub_velocity_command.publish(speed)
            break

        angle_to_goal = atan2(inc_y, inc_x)

        # pokud nesmeruji primo k mistu cile, otoc se smerem k nemu
        if (angle_to_goal - mobile_base.rotation > 0.1 and abs(angle_to_goal - 2*pi - mobile_base.rotation) > abs(angle_to_goal - mobile_base.rotation)) or (abs(angle_to_goal + 2*pi - mobile_base.rotation) < abs(angle_to_goal - mobile_base.rotation) and abs(angle_to_goal - mobile_base.rotation) > 0.1):
            speed.linear.x = 0.0
            speed.angular.z = 0.5
        elif angle_to_goal - mobile_base.rotation < -0.1 or (abs(angle_to_goal - 2*pi - mobile_base.rotation) < abs(angle_to_goal - mobile_base.rotation) and abs(angle_to_goal - mobile_base.rotation) > 0.1):
            speed.linear.x = 0.0
            speed.angular.z = -0.5
        # pokud uz na nej mirim, pak k nemu jedu
        else:
            speed.linear.x = 0.5
            speed.angular.z = 0.0

        pub_velocity_command.publish(speed)
        mobile_base.current_position_and_angle()
        r.sleep()

    # kdyz jsem dojel na misto, zarovnej se na pozadovany finalni uhel
    if not on_place:
        rospy.loginfo("Robot se dostal na pozadovane misto, otaceni do finalniho smeru...")
    final_rotation = radians(final_angle_in_degs)

    while not rospy.is_shutdown():
        # pokud jeste nejsem spravne dotoceny, otoc se do spravneho smeru
        if (final_rotation - mobile_base.rotation > 0.1 and abs(final_rotation - 2*pi - mobile_base.rotation) > abs(final_rotation - mobile_base.rotation)) or (abs(final_rotation + 2*pi - mobile_base.rotation) < abs(final_rotation - mobile_base.rotation) and abs(final_rotation - mobile_base.rotation) > 0.1):
            speed.angular.z = 0.5
        elif final_rotation - mobile_base.rotation < -0.1 or (abs(final_rotation - 2*pi - mobile_base.rotation) < abs(final_rotation - mobile_base.rotation) and abs(final_rotation - mobile_base.rotation) > 0.1):
            speed.angular.z = -0.5
        # pokud jsem v toleranci, prestan se otacet a vytiskni finalni polohu, do ktere robot dojel
        else:
            speed.angular.z = 0.0
            break
        pub_velocity_command.publish(speed)
        mobile_base.current_position_and_angle()
        mobile_base.current_speed()
        r.sleep()
    rospy.loginfo("Robot is facing final direction. Waiting for robot to stop the rotation...")
    while(abs(mobile_base.get_current_speed()[0]) > 0.01 or abs(mobile_base.get_current_speed()[1]) > 0.01):
        speed.angular.z = 0.0
        pub_velocity_command.publish(speed)
        rospy.sleep(0.01)

    mobile_base.final_position_and_angle()


def analyze_picture_with_mediapipe(step=None):
    """
    Funkce porizujici fotku pomoci kamery robota, posila ji na vyhodnoceni pomoci mediapipe 
    a vysledky zapisuje do modelu HumanInfo.
    """
    image_converter.set_look_at_image(True)
    if step:
        rospy.loginfo("Analyza snimku porizeneho v kroku %d ..." % int(step))
    else:
        rospy.loginfo("Probiha analyza snimku z kamery...")
    # dokud se obraz analyzuje, cekame na vysledek
    while(image_converter.get_look_at_image()):
        rospy.sleep(0.001)
    cv_image_as_txt = image_converter.get_last_image_as_txt()
    image_dict = {"image": cv_image_as_txt}
    results = mediapipe_image_service_client.call(image_dict)
    human_info.human_found = results["human_found"]
    human_info.human_centered = results["human_centered"]
    human_info.final_robot_rotation.x = results["final_robot_rotation.x"]
    human_info.final_robot_rotation.y = results["final_robot_rotation.y"]
    human_info.center_location.x = results["human_center_x"]
    human_info.center_location.y = results["human_center_y"]


def analyze_picture_with_alphapose():
    """
    Function that sends current image from robot to AlphaPose image server
    """
    rospy.loginfo("Start of the image analysis via AlphaPose...")
    image_converter.set_look_at_image(True)
    while(image_converter.get_look_at_image()):
        rospy.sleep(0.001)
    cv_image_as_txt = image_converter.get_last_image_as_txt()
    test_dict = {
        "Hello": "AlphaPose",
        "image_name": "human_alphapose.jpg",
        "image_as_txt": cv_image_as_txt,
        "out_path": OUT_PATH,
    }
    # Sending data to AlphaPose image server, waits for response and saves the result
    response = alphapose_image_service_client.call(test_dict)
    human_info.alphapose_json_result = response["json_results"]
    rospy.loginfo("Analysis of pose via AlphaPose finished. Results:")
    print(human_info.alphapose_json_result)


def analyze_picture_with_motionbert():
    """
    Funkce odesilajici podnet ke zpracovani dat obdrzenych z AlphaPose pomoci MotionBERT 
    """
    rospy.loginfo("Zahajeni analyzy obrazu pomoci MotionBERT...")
    test_dict = {
        "Hello": "MotionBERT",
        "out_path": OUT_PATH,
        "json_results": human_info.alphapose_json_result,
    }
    response = motionbert_image_service_client.call(test_dict)
    human_info.motionbert_joints_positions = response["human_joints_positions"]
    rospy.loginfo("Analyza pozy pomoci MotionBERT dokoncena. Vysledna data:")
    rospy.loginfo(human_info.motionbert_joints_positions)
    

def find_human():
    """
    Funkce, ktera ma za ukol nalezt cloveka dle zadaneho sekvence pohybu hlavy a zakladny
    popis kroku:
    step 0: robot poridi snimek 1.
    step 1: robot otoci hlavou do uhlu 45 stupnu, poridi snimek
    step 2: robot otoci hlavou do uhlu -45 stupnu, poridi snimek
    step 3: robot otoci zakladnou do uhlu -135 stupnu, poridi snimek
    step 4: robot otoci hlavou do uhlu 0 stupnu, poridi snimek
    step 5: robot otoci hlavou do uhlu 45 stupnu, poridi snimek
    step 6: robot se otoci zakladnou do uhlu 90 stupnu, poridi snimek.
    step 7: robot otoci hlavou do u uhlu 0 stupnu, poridi snimek.
    step 8: Pokud nenajde cloveka ani ve stepu 7, otoci zakladnou do uhlu 0 a bude opet zacinat krokem 0
    """
    step = 0
    base_rotation = mobile_base.rotation
    # dokud nenajdeme cloveka, hledejme ho 
    while(not human_info.is_found()):

        # pokud ani na poslednim snimku nenajdeme cloveka, pak hledejme znovu od zacatku.
        if step == 8:
            move_base(Point(), degrees(base_rotation), on_place=True)
            step = 0
            # human_found = True # pro testovaci ucely
            # rospy.loginfo("Cyklus hledani cloveka ukoncen")
            rospy.loginfo("End of search cycle, human not found. Starting new cycle...")
            continue
        if step == 0 and (abs(latest_joint_states.return_head_joints_positions()[0]) > 0.01 or abs(latest_joint_states.return_head_joints_positions()[0]) > 0.01):
            head_position = latest_joint_states.return_head_joints_positions()
            base_rotation = base_rotation + head_position[0]
            move_head(head_action_client, 0, 0, wait_for_result=False)
            move_base(Point(), degrees(base_rotation), on_place=True)
        if step == 1:
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
        
        # pockame na zpracovani obrazu.
        analyze_picture_with_mediapipe(step)

        step += 1
    rospy.loginfo("Human found on picture!")


def center_camera_on_human():
    """
    Funkce provadejici zacentrovani na cloveka po jeho nalezeni v obraze
    """
    rospy.loginfo("Centering on human in the picture...")
    [head_1_joint_pos, head_2_joint_pos] = latest_joint_states.return_head_joints_positions()
    final_head_hor_angle = degrees(head_1_joint_pos) + human_info.final_robot_rotation.x
    final_head_ver_angle = degrees(head_2_joint_pos) + human_info.final_robot_rotation.y

    # uvazujeme maximalni uhel otoceni hlavy approx. 70 stupnu
    if abs(final_head_hor_angle) < 70:
        move_head(head_action_client, final_head_hor_angle, final_head_ver_angle)
    elif abs(final_head_hor_angle) >= 70:
        move_head(head_action_client, 0, final_head_ver_angle, wait_for_result=False)
        move_base(Point(), degrees(mobile_base.rotation) + final_head_hor_angle, on_place=True)
    head_joints_positions = latest_joint_states.return_head_joints_positions()
    rospy.loginfo("Centering complete. Current head joints positions:\nhead_1_joint: %s\nhead_2_joint: %s" % (degrees(head_joints_positions[0]), degrees(head_joints_positions[1])))


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


def get_point_on_ground(right_shoulder, right_wrist):
    """
    Funkce ktera ze dvou bodu ruky spocita prusecik primky mezi nimi se zemi
    """
    vector = Location3DJoint(x=right_wrist.x - right_shoulder.x, y=right_wrist.y - right_shoulder.y, z=right_wrist.z - right_shoulder.z)
    print("Vektor mezi klouby:(%f, %f, %f)" %(vector.x, vector.y, vector.z))
    final_point = Location3DJoint(
        x = right_shoulder.x - (right_shoulder.z/vector.z)*vector.x,
        y = right_shoulder.y - (right_shoulder.z/vector.z)*vector.y,
        z = 0
    )
    print("finalni bod:(%f, %f, %f)" %(final_point.x, final_point.y, final_point.z))
    return final_point


def get_rotation_matrix(alpha):
    """
    Funkce vracejici rotacni matici se zadanym uhlem ve 2D
    """
    return np.array([
        [cos(alpha),    -sin(alpha),    0],
        [sin(alpha),    cos(alpha),     0], 
        [0,             0,              1]
    ])


def get_translation_matrix(x, y):
    """
    Funkce vracejici translacni matici do urciteho bodu v rovine
    """
    return np.array([
        [1,    0,    x],
        [0,    1,    y], 
        [0,    0,    1]
    ])


def transform_points_to_odom(final_point):
    """
    Function transforming location of human and the final point to odom coordinate system
    Input: final_point - Location3DJoint() object in MotionBERT human coordinate system
    Output: final_point_matrix - np.array() with coordinates of final point in odom coordinate system
            human_coordinates - np.array() with coordinates of human in odom coordinate system
    """
    final_point_matrix = np.array([[final_point.x], [final_point.y], [1]])
    rot_matrix = get_rotation_matrix(-pi/2)
    final_point_matrix = np.matmul(rot_matrix, final_point_matrix)
    #print("Bod po rotaci do zakladny cloveka")
    #print(final_point_matrix)

    # Distance from robot base to human extracted from depth image
    distance_to_human = human_info.distance_from_robot
    # Human coordinates in human coordinate system -> [0, 0, 1]
    human_coordinates = np.array([[0], [0], [1]])
    trans_matrix = get_translation_matrix(distance_to_human, 0)
    final_point_matrix = np.matmul(trans_matrix, final_point_matrix)
    human_coordinates = np.matmul(trans_matrix, human_coordinates)
    # print("Finalni bod po translaci do zakladny robota")
    # print(final_point_matrix)
    # print("Pozice cloveka po translaci do zakladny robota")
    # print(human_coordinates)

    # Rotation to match the orientation of odom coordinate system.
    base_position_and_rotation = mobile_base.get_current_position_and_rotation()
    head_joint_position = latest_joint_states.return_head_joints_positions()
    trans_matrix = get_translation_matrix(base_position_and_rotation[0], base_position_and_rotation[1])
    rot_matrix = get_rotation_matrix(base_position_and_rotation[2] + head_joint_position[0])
    final_point_matrix = np.matmul(rot_matrix, final_point_matrix)
    human_coordinates = np.matmul(rot_matrix, human_coordinates)
    # print("Finalni bod po rotaci do souradneho systemu odom")
    # print(final_point_matrix)
    # print("Pozice cloveka po rotaci do souradneho systemu odom")
    # print(human_coordinates)

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
    Funkce ktera si zpracuje obdrzene body od MotionBERTa a dostane z nich polohu, kam ma robot dojet.
    """
    rospy.loginfo("Vytvarim si slovnik nalezenych kloubu cloveka...")
    human_info.create_joints_dict()

    rospy.loginfo("Posouvam pocatek souradneho systemu nalezenych bodu...")
    human_info.change_reference_of_joints()

    rospy.loginfo("Hledam prusecik primky se zemi...")
    right_shoulder = human_info.motionbert_joints_positions["RShoulder"]
    right_wrist = human_info.motionbert_joints_positions["RWrist"]
    final_point = get_point_on_ground(right_shoulder, right_wrist)

    rospy.loginfo("Transformuji nalezeny bod do souradneho systemu odom...")
    # prvni je rotace finalniho bodu v miste cloveka.
    [final_point_matrix, human_coordinates] = transform_points_to_odom(final_point)
    
    rospy.loginfo("Vypocet finalniho uhlu otoceni po presunu do pozadovaneho mista...")
    inc_x = human_coordinates[0] - final_point_matrix[0]
    inc_y = human_coordinates[1] - final_point_matrix[1]

    final_angle_in_degs = degrees(atan2(inc_y, inc_x))
    print("Finalni uhel natoceni zakladny je: %f" % final_angle_in_degs)

    rospy.loginfo("Vykonavani pohybu robota do pozadovaneho mista...")
    raw_input("Press ENTER to confirm movement to final point")

    return [final_point_matrix, final_angle_in_degs]


def move_robot_to_final_point(final_point_matrix, final_angle_in_degs):
    """
    Funkce vykonavajici pohyb na nalezeny bod kam clovek ukazuje.
    """

    move_head(head_action_client, 0, 0, wait_for_result=False)
    move_base(Point(x = final_point_matrix[0], y=final_point_matrix[1]), final_angle_in_degs)

if __name__ == '__main__':

    try:
        rospy.init_node("tiago_controller", anonymous=True)

        
        head_action_client = create_action_client("/head_controller/follow_joint_trajectory")
        #hand_left_action_client = create_action_client('/hand_left_controller/follow_joint_trajectory')
        tf_listener = TransformListener()
        mobile_base = BasePositionWithOrientation()
        image_converter = ImageConverter()
        latest_joint_states = LatestJointStates()
        human_info = HumanInfo()
        camera_info = Camera()

        mediapipe_image_service_client = ServiceClient()
        alphapose_image_service_client = ServiceClient(port=242425)
        motionbert_image_service_client = ServiceClient(port=242426)

        pub_velocity_command = rospy.Publisher("/mobile_base_controller/cmd_vel", Twist, queue_size=1)

        args = parse_args()

        if args.demo == "camera":
            camera_info = Camera()
            """ image_converter.lock.acquire()
            image_converter.look_at_camera_info = True
            image_converter.look_at_depth_image = True
            image_converter.lock.release() """
            #image_converter.look_at_depth_image = True
            analyze_picture_with_mediapipe()
            camera_info.find_distance_to_human(human_info.center_location)

        elif args.demo == "move_head":
            move_head(head_action_client, 0, 0)
            move_head(head_action_client, 30, 0)
            move_head(head_action_client, -30, 0)
            move_head(head_action_client, 0, 0)
            move_head(head_action_client, 0, 30)
            move_head(head_action_client, 0, -30)
            move_head(head_action_client, 0, 0)

        elif args.demo == "rotate_base":
            current_base_position = mobile_base.get_current_position_and_rotation()
            while(current_base_position[2] == None):
                rospy.sleep(0.1)
                current_base_position = mobile_base.get_current_position_and_rotation()

            move_base(Point(), degrees(current_base_position[2])+45, on_place=True)
            move_base(Point(), degrees(current_base_position[2])-45, on_place=True)
            move_base(Point(), degrees(current_base_position[2]), on_place=True)

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
            # dokud nebudu chtit ukoncit ovladani robota, opakuj pracovni smycku
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
                    camera_info.find_distance_to_human(human_info.center_location)
                    rospy.loginfo("Human is in center of the camera view. Press ENTER to regognise pose and move robot...")
                    i, o, e = select.select( [sys.stdin], [], [], 3)
                    if (i):
                        sys.stdin.readline().strip() # vymaze stisknuty ENTER pro zamezeni opetovneho rozpoznavani
                        image_converter.get_current_view(show=False, save=True, filename="human.jpg")
                        analyze_picture_with_alphapose()
                        analyze_picture_with_motionbert()
                        [final_point_matrix, final_angle_in_degs] = find_final_point()
                        move_robot_to_final_point(final_point_matrix, final_angle_in_degs)
                    else:
                        rospy.loginfo("Probiha opetovna kontrola zacentrovani na cloveka...")
                elif human_info.is_found() and human_info.is_centered() and args.demo == "watch_human":
                    tries = 0
                    rospy.loginfo("Human is in center of the camera view.")
                    rospy.sleep(0.5)
                    rospy.loginfo("Probiha opetovna kontrola zacentrovani na cloveka...")
                analyze_picture_with_mediapipe()

        rospy.spin()
        
    except rospy.ROSInterruptException as e:
        print("Neco se pokazilo: %s" % e)