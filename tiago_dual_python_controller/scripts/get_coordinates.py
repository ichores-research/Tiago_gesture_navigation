#!/usr/bin/env python3
import sys
import rospy
import cv2
import mediapipe.python as mp
import numpy as np
import math
from std_msgs.msg import String


ROBOT_HEIGHT = 1.126886 # hodnota vysky hlavy robota v zakladni poloze ze simulace
HUMAN_ARM_BODY_RATIO = 0.6

class LocationPoint:
  def __init__(self, x, y):
    self.x = x
    self.y = y

def init_mediapipe():
    mp_drawing = mp.solutions.drawing_utils
    #mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    return mp_drawing, mp_pose # mp_drawing_styles, mp_pose

def recognise_person_from_image(mp_drawing, mp_pose, image_file): #, mp_drawing_styles
    """
    Funkce, ktera rozpozna pozu cloveka a navrati souradnice jeho bodu v obdrzenem obraze
    Souradnice v obraze jsou pocitany od leveho horniho rohu tak, ze sirka obrazu je souradnice x, vyska obrazu je souradnice y
    """
    # For static images:
    with mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5) as pose:

        image = cv2.imread("/home/guest/tiago_dual_public_ws/src/python_control_testing_functions/images/" + image_file) 
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # vypis si souradnice zajimavych bodu
        if not results.pose_landmarks:
            print("Picture process error, skipping picture %s" % image_file)
            return None
        print("Picture: %s" % image_file)
        
        # definice pomocnych bodu
        approximate_base_location = LocationPoint(0,0)
        right_wrist_from_base = LocationPoint(0,0)
        
        # zpracovani nalezene pozice na souradnice a uhly nakloneni pro robota, spolu s pribliznym meritkem podle velikosti osoby z obrazku
        human_height, rot_y, arm_length, coord_x, rot_z = process_returned_points(results, approximate_base_location, right_wrist_from_base)
        print("Relative human height: %f" % human_height)
        print("Arm length: %f" % arm_length)
        print("Arm length to human height ratio: %f" % (float(arm_length)/float(human_height)))
        print("Angle of index finger in y-axis: %f" % rot_y)
        scale = ROBOT_HEIGHT / human_height
        print("Scale: %f" % scale)

        # vytisteni nalezenych dulezitych bodu
        wrist_position_and_angle_msg = ",".join([str(coord_x), str(-right_wrist_from_base.x * 0.8*scale), str(-right_wrist_from_base.y*scale), str(rot_y), str(rot_z)])
        print("Wrist position message: %s" % wrist_position_and_angle_msg)

        # Draw pose landmarks on the image.
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS)
        cv2.imwrite('/home/guest/tiago_dual_public_ws/src/python_control_testing_functions/tmp/' + image_file, annotated_image)

    return wrist_position_and_angle_msg

def process_returned_points(results, approximate_base_location, right_wrist_from_base):
    """
    Funkce ktera z navracenych souradnic obrazovych bodu udela souradnice a natoceni pro ruku robota
    """
    # prejmenuj si dulezite parametry pro vypocty
    left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    right_index = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
    right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

    # priblizna lokace zakladny robota je mezi koleny a boky cloveka
    approximate_base_location.x = (left_knee.x + right_knee.x + left_hip.x + right_hip.x)/4 
    approximate_base_location.y = (left_knee.y + right_knee.y + left_hip.y + right_hip.y)/4

    # vypoctene souradnice zapesti cloveka od pomyslne zakladny robota, odpovidajici poloze end efektoru robota 
    right_wrist_from_base.x = right_wrist.x - approximate_base_location.x
    right_wrist_from_base.y = right_wrist.y - approximate_base_location.y

    arm_length = (math.sqrt(math.pow(right_wrist.x - right_elbow.x, 2) + math.pow(right_wrist.y - right_elbow.y, 2)) +
                    math.sqrt(math.pow(right_elbow.x - right_shoulder.x, 2) + math.pow(right_elbow.y - right_shoulder.y, 2)))

    # vypocet vysku cloveka pro korekci ruzne vysky lidi
    human_height = math.sqrt(math.pow(nose.x - approximate_base_location.x, 2) + math.pow(nose.y - approximate_base_location.y, 2))

    # vypocet naklonu ukazovacku v obraze
    rot_y = math.degrees(math.atan2(right_elbow.y - right_index.y,right_elbow.x - right_index.x))

    scale = ROBOT_HEIGHT/human_height

    # uprav souradnici x pro robota v zavislosti velikosti souradnice y, zatim jednoduchy kruh
    coord_y = -right_wrist_from_base.x * 0.7*scale
    coord_x = 0
    if coord_y > 0.65:
        coord_x = 0
    else:
        coord_x = math.sqrt(math.pow(0.65, 2) - math.pow(coord_y, 2))

    # pokud je ruka na obrazku nestandartne mala, pravdepodobne smeruje k nam, pridej na souradnici x
    arm_body_ratio = arm_length/human_height
    if arm_body_ratio < HUMAN_ARM_BODY_RATIO:
        coord_x = coord_x + HUMAN_ARM_BODY_RATIO - arm_body_ratio

    rot_z = math.degrees(math.atan2(coord_y, coord_x))
    """ if coord_y >= 0.65:
        rot_z = 90
    elif coord_y < 0.65 and coord_y >= -0.25:
        rot_z = 180/0.9*coord_y - (18/0.9) """

    return human_height, rot_y, arm_length, coord_x, rot_z

def talker(pub, rate, wrist_position_and_angle_msg):
    """
    Talker ktery vysila zpravu do kanalu chatter s daty pro pohyb robota
    """
    for i in range(1):
        rospy.loginfo(wrist_position_and_angle_msg)
        pub.publish(wrist_position_and_angle_msg)
        rate.sleep()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

if __name__ == '__main__':
    mp_drawing, mp_pose =  init_mediapipe()
    IMAGE_FILES = ["image_2.jpg"] # "test_image_man_1.jpg", "test_image_woman_1.jpg", "image_1.jpg", "image_2.jpg", "image_3.jpg", "tpose.jpg"
    pub = rospy.Publisher('tiago_left_wrist_position', String, queue_size=10)
    rospy.init_node('tiago_left_wrist_position_talker', anonymous=True)
    rate = rospy.Rate(0.2) # 10 = 10hz
    for image_file in IMAGE_FILES:
        wrist_position_and_angle_msg = recognise_person_from_image(mp_drawing, mp_pose, image_file)
        if wrist_position_and_angle_msg is not None:
            try:
                talker(pub, rate, wrist_position_and_angle_msg)
            except rospy.ROSInterruptException:
                pass