#!/usr/bin/env python
import rospy
import cv2

from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String

from sensor_msgs.msg import Image, CameraInfo

bridge = CvBridge()

def image_callback(msg):
    print("Recieved an image")
    print((msg.width, msg.height))
    rospy.sleep(1)
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('camera_info_listener', anonymous=True)

    rospy.Subscriber("/xtion/rgb/camera_info", CameraInfo, image_callback)

    # spin() simply keeps python from exiting until this node is stopped by ctrl + c
    rospy.spin()

if __name__ == '__main__':
    listener()