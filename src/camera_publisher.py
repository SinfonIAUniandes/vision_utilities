#!/usr/bin/env python
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import constants

def main():
    rospy.init_node(constants.WEBCAM_PUBLISHER_NAME, anonymous=True)
    pub = rospy.Publisher(constants.LOCAL_FRONT_CAMERA, Image, queue_size=10)
    bridge = CvBridge()
    cap = cv2.VideoCapture(0)

    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            continue
        msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        pub.publish(msg)
        rate.sleep()

    cap.release()

if __name__ == '__main__':
    main()
