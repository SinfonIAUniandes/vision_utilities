import time
from threading import Event
from typing import Callable, Optional, TypeVar

import rospy
from cv2.typing import MatLike
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from utils.load_balancer import LoadBalancer
from utils.singleton import Singleton

T = TypeVar("T")


@Singleton
class CameraTopic:
    load_balancer: LoadBalancer[MatLike]
    bridge = CvBridge()

    def __init__(self, camera_name: str):
        self.camera_name = camera_name
        self.load_balancer = LoadBalancer[MatLike]()
        self._image: Optional[MatLike] = None
        rospy.Subscriber(self.camera_name, Image, self.camera_subscriber)

    def subscribe(
        self, callback: Callable[[MatLike], None], wait_turns: int = 1
    ) -> int:
        sid = self.load_balancer.subscribe(callback, wait_turns_for_call=wait_turns)
        self.load_balancer.rebuild_schedule()
        return sid

    def unsubscribe(self, subscriber_id: int) -> None:
        self.load_balancer.unsubscribe(subscriber_id)
        self.load_balancer.rebuild_schedule()

    def get_image(self) -> Optional[MatLike]:
        return self._image

    def process_until(
        self,
        processor: Callable[[MatLike], Optional[T]],
        timeout: float,
        wait_turns: int = 1,
    ) -> Optional[T]:
        """
        Process frames until processor returns a non-None value or timeout.

        Args:
            processor: Function that receives a frame and returns a result or None.
            timeout: Maximum time in seconds to wait for a result.
            wait_turns: Frame interval for the load balancer.

        Returns:
            The first non-None result from processor, or None on timeout.
        """
        result: Optional[T] = None
        done = Event()

        def frame_handler(image: MatLike):
            nonlocal result
            if done.is_set():
                return
            value = processor(image)
            if value is not None:
                result = value
                done.set()

        sid = self.subscribe(frame_handler, wait_turns=wait_turns)
        done.wait(timeout=timeout)
        self.unsubscribe(sid)

        return result

    def camera_subscriber(self, msg: Image):
        self._image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.load_balancer.run_turn(self._image)


__all__ = ["CameraTopic"]
