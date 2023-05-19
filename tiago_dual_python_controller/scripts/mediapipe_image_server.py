#!/usr/bin/env python3

"""
Skript slouzici jako server zpracovavajici obrazova data z robota pomoci MediaPipe
"""
import sys
import rospy
import cv2
import mediapipe.python as mp
import numpy as np
import magic
import base64
import argparse

from std_msgs.msg import String
from sensor_msgs.msg import Image

from image_share_service.service_server import ServiceServer
from classes import LocationPoint, HumanInfo

OUT_PATH = "/home/guest/image_recog_results/"


ROBOT_HEIGHT = 1.126886 # hodnota vysky hlavy robota v zakladni poloze ze simulace   


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=bool, default=False, help='emables saving of partial results from MediaPipe')
    opts = parser.parse_args()
    return opts


def init_mediapipe():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    return mp_drawing, mp_pose

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
    human_info.distance_from_robot = None
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

        if args.debug:
            # Draw pose landmarks on the image.
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS)
            cv2.imwrite(OUT_PATH + "vis/human_mediapipe.jpg", annotated_image)

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

    rospy.loginfo("Rotations needed to center on human:\nhor_rotation = %f\nver_rotation = %f" % (human_info.final_robot_rotation.x, human_info.final_robot_rotation.y))

    return 


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
        "human_center_x": human_info.center_location.x,
        "human_center_y": human_info.center_location.y,
        "final_robot_rotation.x": human_info.final_robot_rotation.x,
        "final_robot_rotation.y": human_info.final_robot_rotation.y,
        "Hello": "ROS",
    }
    rospy.loginfo("MediaPipe recognition complete. Waiting for another picture...")
    return results_dict

if __name__ == '__main__':
    mp_drawing, mp_pose =  init_mediapipe()

    args = parse_args()

    rospy.init_node('human_image_position_talker', anonymous=True)
    pub_human_info = rospy.Publisher('human_info_cmd', String, queue_size=10)
    human_info = HumanInfo()
    image_server = ServiceServer(callback_image_server)
    rospy.loginfo("Image server running. waiting for imput image...")

    rospy.spin()
    