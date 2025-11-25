#!/usr/bin/env python3
"""ROS service node that calls Ollama to answer a prompt and image.

Service: `/vision_utilities/recognition/vlm_srv`
Request: `prompt` (string), `temperature` (float64)
Response: `answer` (string)

This node subscribes to a camera topic, keeps the latest frame, and when the
service is called it encodes the latest image (if available) and sends it to
Ollama together with the prompt and temperature.

The Ollama model can be overridden via the private ROS param `~model`
(default: `gemma3_12b`).
"""
import base64
import rospy
import ollama

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from perception_msgs.srv import vlm_srvRequest, vlm_srvResponse, vlm_srv
import cv2
from typing import Optional


class OllamaVLMService:
	bridge = CvBridge()
	image = None
	active = False

	def __init__(self, camera_topic: str):
		self.active = True
		self.image = None
		self.model = rospy.get_param("~model", "gemma3_12b")
		# Service to receive prompt and temperature
		self.service = rospy.Service("/vision_utilities/recognition/vlm_srv", vlm_srv, self.handle_vlm)
		# Subscribe to camera feed
		rospy.Subscriber(camera_topic, Image, self.camera_subscriber)
		rospy.loginfo(f"OllamaVLMService initialized model={self.model} camera={camera_topic}")

	def camera_subscriber(self, msg: Image):
		"""Keep latest camera frame as BGR numpy array."""
		try:
			self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
		except Exception as e:
			rospy.logerr(f"Failed to convert image message: {e}")
			self.image = None

	def encode_image_base64(self, img) -> Optional[str]:
		"""Encode an OpenCV BGR image to base64 JPEG string for Ollama.

		Returns None if encoding fails or image is None.
		"""
		if img is None:
			return None
		ret, buf = cv2.imencode('.jpg', img)
		if not ret:
			rospy.logerr("cv2.imencode failed")
			return None
		return base64.b64encode(buf.tobytes()).decode('utf-8')

	def call_ollama_model(self, prompt: str, temperature: float, image_b64: Optional[str] = None) -> str:
		"""Call Ollama chat API with prompt and optional image, return text answer."""
		try:
			messages = [{"role": "user", "content": prompt}]
			if image_b64:
				# Ollama expects images as base64 strings inside an `images` field
				messages[0]["images"] = [image_b64]

			response = ollama.chat(
				model=self.model,
				messages=messages,
			)

			return response['message']['content']
		except Exception as e:
			rospy.logerr(f"Error calling Ollama: {e}")
			return f"<error> {e}"

	def handle_vlm(self, req: vlm_srvRequest) -> vlm_srvResponse:
		"""ROS service handler: receives a prompt and temperature, returns answer."""
		rospy.loginfo(f"VLM request received: prompt='{req.prompt[:80]}' temperature={req.temperature}")

		image_b64 = self.encode_image_base64(self.image)
		if image_b64 is None:
			rospy.logwarn("No camera image available; calling VLM with prompt only")

		answer_text = self.call_ollama_model(req.prompt, req.temperature, image_b64)

		resp = vlm_srvResponse()
		resp.answer = answer_text
		rospy.loginfo("VLM response ready")
		return resp