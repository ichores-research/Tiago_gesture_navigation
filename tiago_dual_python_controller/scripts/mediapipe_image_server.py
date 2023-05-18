#!/usr/bin/env python3
import sys
import rospy
import cv2
import mediapipe.python as mp
import numpy as np
import magic
import base64

from std_msgs.msg import String
from sensor_msgs.msg import Image

from image_share_service.service_server import ServiceServer
from classes import LocationPoint, HumanInfo


ROBOT_HEIGHT = 1.126886 # hodnota vysky hlavy robota v zakladni poloze ze simulace
HUMAN_ARM_BODY_RATIO = 0.6

mp_drawing = None
mp_pose = None    

def init_mediapipe():
    mp_drawing = mp.solutions.drawing_utils
    #mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    return mp_drawing, mp_pose # mp_drawing_styles, mp_pose

def find_human_in_image_by_mediapipe(human_info, cv2_img):
    """
    Funkce ktera za pomoci MediaPipe prohleda predlozeny soubor.
    Upravuje tridu HumanInfo s parametry k hledani cloveka v obraze, slouzi k centrovani kamery na cloveka
    Pred kazdym zpracovanim obrazu se vynuluji parametry tridy. 
    """
    human_info.human_found = False
    human_info.human_centered = False
    human_info.center_location.x = None
    human_info.center_location.y = None
    human_info.final_robot_rotation.x = None
    human_info.final_robot_rotation.y = None
    with mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5) as pose:

        image = cv2_img
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # pokud nenaleznu cloveka, vratim ze nebyl nalezen
        if not results.pose_landmarks:
            rospy.loginfo("Human not found in the picture")
            return 
        
        # pokud ho najdu, vypisu jeho souradnice do zpravy, kterou pak poslu
        rospy.loginfo("Human found in image")
        human_info.human_found = True
        get_human_center_location(results, human_info)
        get_needed_robot_rotation(human_info)
        return

def get_human_center_location(results, human_info):
    """
    Funkce ktera z navracenych souradnic obrazovych bodu udela souradnice a natoceni pro ruku robota
    """
    # prejmenuj si dulezite parametry pro vypocty
    left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

    # Vypocti teziste zadanych bodu
    human_info.center_location.x = (left_knee.x + right_knee.x + left_shoulder.x + right_shoulder.x)/4
    human_info.center_location.y = (left_knee.y + right_knee.y + left_shoulder.y + right_shoulder.y)/4

    rospy.loginfo("Center of human in image is:\nx = %f\ny = %f" % (human_info.center_location.x, human_info.center_location.y))

    if abs(human_info.center_location.x - 0.5) < 0.05 and abs(human_info.center_location.y - 0.5) < 0.05:
        human_info.human_centered = True

    return

def get_needed_robot_rotation(human_info):
    """
    Funkce vypocitavajici uhel natoceni hlavy ci zakladny potrebny k zarovnani na stred cloveka.
    """
    if human_info.is_centered():
        rospy.loginfo("Human in picture is in center, no need to move")
        return

    pos_x = human_info.center_location.x - 0.5
    pos_y = human_info.center_location.y - 0.5

    human_info.final_robot_rotation.x = (-pos_x*(60))
    human_info.final_robot_rotation.y = (-pos_y*(49.5))

    print("Rotations needed to center on human:\nhor_rotation = %f\nver_rotation = %f" % (human_info.final_robot_rotation.x, human_info.final_robot_rotation.y))

    return 


def create_human_info_msg(human_info):
    """
    Funkce vytvarejici string pro odeslani do hlavniho skriptu, ktery obsahuje informace o poloze cloveka a potrebne rotaci
    Format zpravy je nasledujici:
    "human_info.human_found,human_info.human_centered,human_info.final_robot_rotation.x,human_info.final_robot_rotation.y"
    """
    humam_info_message = ",".join([str(human_info.is_found()), str(human_info.is_centered()), str(human_info.final_robot_rotation.x), str(human_info.final_robot_rotation.y)])
    return humam_info_message

def str_msg_publisher(pub, str_msg):
    """
    Talker ktery vysila zpravu do kanalu chatter s daty pro pohyb robota
    """
    rospy.loginfo("Odesilana zprava je:\n%s" % str_msg)
    pub.publish(str_msg)

def callback_image_server(stuff:dict):
    """
    Callback ktery obdrzi slovnik s cv_image k analyze, vrati take slovnik s parametry pro human_info
    """
    rospy.loginfo("Image received, processing...")
    image_as_txt = stuff["image"]
    jpg_original = base64.b64decode(image_as_txt)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    image_buffer = cv2.imdecode(jpg_as_np, flags=1)
    find_human_in_image_by_mediapipe(human_info, image_buffer)
    results_dict = {
        "human_found": human_info.human_found,
        "human_centered": human_info.human_centered,
        "final_robot_rotation.x": human_info.final_robot_rotation.x,
        "final_robot_rotation.y": human_info.final_robot_rotation.y,
        "Hello": "ROS",
    }
    rospy.loginfo("MediaPipe recognition complete. Waiting for another picture...")
    return results_dict

if __name__ == '__main__':
    mp_drawing, mp_pose =  init_mediapipe()

    #pub = rospy.Publisher('tiago_left_wrist_position', String, queue_size=10)
    rospy.init_node('human_image_position_talker', anonymous=True)
    pub_human_info = rospy.Publisher('human_info_cmd', String, queue_size=10)
    #IMAGES = ["step0.jpg", "step1.jpg", "step2.jpg", "step3.jpg", "step4.jpg", "step5.jpg", "step6.jpg", "step7.jpg"]
    human_info = HumanInfo()
    image_server = ServiceServer(callback_image_server)
    rospy.loginfo("Image server running. waiting for imput image...")
    """ for image_file in IMAGES:
        rospy.loginfo("Processing image file: %s" % image_file)
        cv2_img = cv2.imread('/home/guest/tiago_dual_public_ws/src/python_control_testing_functions/tmp/' + image_file)
        find_human_in_image_by_mediapipe(mp_pose, cv2_img, human_info)
        human_info_msg = create_human_info_msg(human_info)
        str_msg_publisher(pub_human_info, human_info_msg)
    rospy.spin() 
    """
    rospy.spin()
    

        