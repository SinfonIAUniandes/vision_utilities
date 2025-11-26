#!/usr/bin/env python3
import base64
import rospy
import ollama
from openai import AzureOpenAI

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from perception_msgs.srv import vlm_srvRequest, vlm_srvResponse, vlm_srv
import cv2
from typing import Optional
import os


class VLMService:
	bridge = CvBridge()
	image = None
	active = False

	# (removed duplicate simple constructor)

	def __init__(self, camera_topic: str, llm_mode: str = 'ollama', model: str = 'gemma3_12b', max_tokens: int = 500):
		self.active = True
		self.image = None
		# Defaults
		self.llm_mode = llm_mode
		self.model = model
		self.max_tokens = max_tokens
		
		if self.llm_mode.lower() == 'openai':
			self.clientGPT = AzureOpenAI(
                azure_endpoint="https://sinfonia.openai.azure.com/",
                api_key=os.getenv("GPT_API"),
                api_version="2024-02-01",
            )

		# Service and subscriber
		self.service = rospy.Service("/vision_utilities/recognition/vlm_srv", vlm_srv, self.handle_vlm)
		rospy.Subscriber(camera_topic, Image, self.camera_subscriber)
		rospy.loginfo(f"VLMService initialized llm_mode={self.llm_mode} model={self.model} camera={camera_topic}")

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


	def call_openai_model(self, prompt: str, temperature: float, image_b64: Optional[str] = None, model: Optional[str] = None) -> str:
		"""Call OpenAI Chat API with prompt and optional image (base64), return text answer.

		Note: OpenAI's chat endpoint does not accept raw base64 images in a dedicated field
		in the same way Ollama does. To keep parity with the existing behavior we pass
		the base64-encoded image inline as an additional user message labeled clearly
		as an image. This keeps the current functionality without replacing it.

		Args:
			prompt: The user prompt to send to the model.
			temperature: Sampling temperature.
			image_b64: Optional base64-encoded JPEG image string.
			model: Optional model name (defaults to 'gpt-4o' if not provided).
		Returns:
			The model's text response, or an error string prefixed with '<error>'.
		"""
		try:
			messages = [
				{
					"role": "user",
					"content": 
					[
						{"type": "text", "text": prompt},
						{
							"type": "image_url", 
							"image_url": {
							"url": f"data:image/png;base64,{image_b64}"
							}
						}
					]
				}
			]

			answer = self.clientGPT.chat.completions.create(
				model=model,
				messages=messages,
				temperature=temperature,
				max_tokens=self.max_tokens,
			)

			return answer.choices[0].message.content
		except Exception as e:
			rospy.logerr(f"Error calling OpenAI ChatCompletion: {e}")
			return f"<error> {e}"

	def handle_vlm(self, req: vlm_srvRequest) -> vlm_srvResponse:
		"""ROS service handler: receives a prompt and temperature, returns answer."""
		rospy.loginfo(f"VLM request received: prompt='{req.prompt[:80]}' temperature={req.temperature}")

		image_b64 = self.encode_image_base64(self.image)
		if image_b64 is None:
			rospy.logwarn("No camera image available; calling VLM with prompt only")

		# Choose backend based on configuration
		llm = (getattr(self, 'llm_mode', 'ollama') or 'ollama').lower()
		if llm == 'openai':
			answer_text = self.call_openai_model(req.prompt, req.temperature, image_b64, model=self.model)
		else:
			answer_text = self.call_ollama_model(req.prompt, req.temperature, image_b64)

		resp = vlm_srvResponse()
		resp.answer = answer_text
		rospy.loginfo("VLM response ready")
		return resp