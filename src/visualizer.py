#!/usr/bin/env python
import rospy
import cv2
from sensor_msgs.msg import Image
from perception_msgs.msg import Polygon
from perception_msgs.srv import visualize_polygon_topic_srv, visualize_polygon_topic_srvRequest, visualize_polygon_topic_srvResponse
from cv_bridge import CvBridge
import numpy as np

import constants

class PolygonRenderer:
    bridge = CvBridge()
    topics = {}

    def __init__(self):
        rospy.init_node(constants.POLYGON_RENDERING_NAME, anonymous=True)
        rospy.Service(constants.SERVICE_RENDER_POLYGON_TOPIC, visualize_polygon_topic_srv, self.service_callback)
        rospy.Subscriber(constants.PEPPER_FRONT_CAMERA, Image, self.camera_callback)
        self.publisher = rospy.Publisher(constants.TOPIC_POLYGON_RENDERER, Image)
        rospy.spin()


    def camera_callback(self, msg: Image):
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        for topic in self.topics:
            polygons = np.array(self.topics[topic]["last"])
            polygons[::2] *= image.shape[1]
            polygons[1::2] *= image.shape[0]
            polygons = np.round(polygons)

            for i in range(0, len(polygons), 4):
               #print(polygons[i:i+2])
               cv2.line(image, (int(polygons[i]), int(polygons[i+1])), (int(polygons[i + 2]), int(polygons[i + 3])), color=(0, 255, 0), thickness=2)

        self.publisher.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))


    def polygon_callback(self, pol: Polygon, topic: str):
        self.topics[topic]["last"] = pol.polygon


    def service_callback(self, request: visualize_polygon_topic_srvRequest):
        available_topics = list(map(lambda x: x[0], filter(lambda x: x[1] == "perception_msgs/Polygon" , rospy.get_published_topics())))
        response = visualize_polygon_topic_srvResponse()

        if request.polygon_topic_name not in available_topics:
            response.visualizing = False
            return response
        
        sub = rospy.Subscriber(request.polygon_topic_name, Polygon, self.polygon_callback, callback_args=request.polygon_topic_name)
        self.topics[request.polygon_topic_name] = {"sub": sub, "last": None}
        response.visualizing = True
        return response


if __name__ == '__main__':
    PolygonRenderer()
