#!/usr/bin/env python3

"""
MediaPipe image server processing image data received from robot.
"""
import sys
import rospy
import cv2
import mediapipe.python as mp
import numpy as np
import base64
import argparse

from std_msgs.msg import String
from sensor_msgs.msg import Image

from image_share_service.service_server import ServiceServer
from classes import LocationPoint, HumanInfo

OUT_PATH = "/home/guest/image_recog_results/" 

CENTER_LOCATION_TOLERANCE = 0.05
FOW_X = 60
FOW_Y = 49.5

def parse_args():
    """
    Function for parsing input argument
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=bool, default=False, help='emables saving of partial results from MediaPipe')
    opts = parser.parse_args()
    return opts


def init_mediapipe():
    """
    Function initializing MediaPipe utils
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    return mp_drawing, mp_pose

def find_human_in_image_by_mediapipe(human_info, cv2_img):
    """
    Function that processes given image with MediaPipe.
    Modifies HumanInfo() object when human is found on the picture
    Before every processing of image we reset the HumanInfo() object
    Input:
        - human_info: HumanInfo() object
        - cv2_img: received image converted back to cv2 image format
    Result:
        Human center location is saved in the HumanInfo() object
    """
    human_info.human_found = False
    human_info.human_centered = False
    human_info.center_location.x = None
    human_info.center_location.y = None
    human_info.distance_from_robot = None
    human_info.final_robot_rotation.x = None
    human_info.final_robot_rotation.y = None

    # Loading solution to find human pose on the image
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    image = cv2_img

    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # If there was no human, no further processing is needed
    if not results.pose_landmarks:
        rospy.loginfo("Human not found in the image!")
        return 
    
    # If human was found, find its center and needed rotations of head to center on human
    rospy.loginfo("Human found in the image!")
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


def get_human_center_location(results, human_info):
    """
    Function computing approximate center of human found in the image from the knees and shoulder joint coordinates.
    Inputs:
        - results: results of analysing image by MediaPipe, expressed in relative image cordinates
        - human_info: HumanInfo() object
    Result:
        Center of human body is located and writen to HumanInfo() object
    """
    
    left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]


    human_info.center_location.x = (left_knee.x + right_knee.x + left_shoulder.x + right_shoulder.x)/4
    human_info.center_location.y = (left_knee.y + right_knee.y + left_shoulder.y + right_shoulder.y)/4

    rospy.loginfo("Center of human in image is:\nx = %f\ny = %f" 
            % (human_info.center_location.x, human_info.center_location.y))

    if (abs(human_info.center_location.x - 0.5) < CENTER_LOCATION_TOLERANCE 
            and abs(human_info.center_location.y - 0.5) < CENTER_LOCATION_TOLERANCE):
        human_info.human_centered = True


def get_needed_robot_rotation(human_info):
    """
    Function computing angle of head rotation needed to center camera on human
    Input:
        - human_info: HumanInfo() object
    """
    if human_info.is_centered():
        rospy.loginfo("Human in picture is in center, no need to move.")
        return

    pos_x = human_info.center_location.x - 0.5
    pos_y = human_info.center_location.y - 0.5

    human_info.final_robot_rotation.x = (-pos_x*(FOW_X))
    human_info.final_robot_rotation.y = (-pos_y*(FOW_Y))

    rospy.loginfo("Rotations needed to center on human:\nhor_rotation = %f\nver_rotation = %f" 
            % (human_info.final_robot_rotation.x, human_info.final_robot_rotation.y))


def callback_image_server(stuff:dict):
    """
    Callback that receives data dictionary with cv_image data, returns distionary with analysis results
    """
    rospy.loginfo("Image received, processing...")
    rospy.loginfo("Main sends: Hello " + stuff['Hello'])
    image_as_txt = stuff["image"]

    # Decode received image from text to cv_image
    jpg_original = base64.b64decode(image_as_txt)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    image_buffer = cv2.imdecode(jpg_as_np, flags=1)

    # Process the image
    find_human_in_image_by_mediapipe(human_info, image_buffer)

    # Return results back to client
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

    rospy.init_node('human_image_position_talker', anonymous=True)
    mp_drawing, mp_pose =  init_mediapipe()

    args = parse_args()
    human_info = HumanInfo()
    
    image_server = ServiceServer(callback_image_server)
    rospy.loginfo("Image processing server with MediaPipe is ready for images...")

    rospy.spin()
    