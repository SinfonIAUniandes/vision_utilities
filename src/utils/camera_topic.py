from pyparsing import Callable
from utils.load_balancer import LoadBalancer
from utils.singleton import Singleton
from std_msgs.msg import Image
from cv_bridge import CvBridge

from cv2.types import MatLike

import rospy

@Singleton
class CameraHandler:
    image = None
    load_balancer: LoadBalancer[MatLike]
    bridge = CvBridge()

    def __init__(self, camera_name: str):
        self.camera_name = camera_name
        self.load_balancer = LoadBalancer[MatLike]()
        self.image = None
        rospy.Subscriber(self.camera_name, Image, self.camera_subscriber)

    def subscribe(self, callback: Callable[[MatLike], None], wait_turns: int = 1) -> int:
        """
        Subscribe to the camera feed and process images with the provided callback.
        """
        sid = self.load_balancer.subscribe(callback, wait_turns_for_call=wait_turns)
        self.load_balancer.rebuild_schedule()
        return sid
    

    def unsubscribe(self, subscriber_id: int) -> None:
        """
        Unsubscribe from the camera feed using the subscriber ID.
        """
        self.load_balancer.unsubscribe(subscriber_id)
        self.load_balancer.rebuild_schedule()


    def camera_subscriber(self, msg: Image):
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.load_balancer.run_turn(self.image)