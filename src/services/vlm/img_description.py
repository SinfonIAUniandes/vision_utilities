#!/usr/bin/env python3
import base64
import os
from typing import Optional

import cv2
import ollama
import rospy
from cv2.typing import MatLike
from openai import AzureOpenAI
from perception_msgs.srv import vlm_srv, vlm_srvRequest, vlm_srvResponse

from utils.camera_topic import CameraTopic


class VLMService:
    def __init__(
        self,
        camera_topic: str,
        llm_mode: str = "ollama",
        model: str = "gemma3_12b",
        max_tokens: int = 500,
    ):
        self.llm_mode = llm_mode
        self.model = model
        self.max_tokens = max_tokens

        if self.llm_mode.lower() == "openai":
            self.clientGPT = AzureOpenAI(
                azure_endpoint="https://sinfonia.openai.azure.com/",
                api_key=os.getenv("GPT_API"),
                api_version="2024-02-01",
            )

        self.camera = CameraTopic(camera_topic)
        self.service = rospy.Service(
            "/vision_utilities/recognition/vlm_srv", vlm_srv, self.handle_vlm
        )
        rospy.loginfo(
            f"VLMService initialized llm_mode={self.llm_mode} model={self.model} camera={camera_topic}"
        )

    def encode_image_base64(self, img: Optional[MatLike]) -> Optional[str]:
        if img is None:
            return None
        ret, buf = cv2.imencode(".jpg", img)
        if not ret:
            rospy.logerr("cv2.imencode failed")
            return None
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def call_ollama_model(
        self, prompt: str, temperature: float, image_b64: Optional[str] = None
    ) -> str:
        try:
            messages = [{"role": "user", "content": prompt}]
            if image_b64:
                messages[0]["images"] = [image_b64]

            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={"temperature": temperature},
            )
            return response["message"]["content"]
        except Exception as e:
            rospy.logerr(f"Error calling Ollama: {e}")
            return f"<error> {e}"

    def call_openai_model(
        self,
        prompt: str,
        temperature: float,
        image_b64: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        try:
            content = [{"type": "text", "text": prompt}]
            if image_b64:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    }
                )

            messages = [{"role": "user", "content": content}]

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
        rospy.loginfo(
            f"VLM request received: prompt='{req.prompt[:80]}' temperature={req.temperature}"
        )

        image = self.camera.get_image()
        image_b64 = self.encode_image_base64(image)
        if image_b64 is None:
            rospy.logwarn("No camera image available; calling VLM with prompt only")

        llm = (getattr(self, "llm_mode", "ollama") or "ollama").lower()
        if llm == "openai":
            answer_text = self.call_openai_model(
                req.prompt, req.temperature, image_b64, model=self.model
            )
        else:
            answer_text = self.call_ollama_model(req.prompt, req.temperature, image_b64)

        resp = vlm_srvResponse()
        resp.answer = answer_text
        rospy.loginfo("VLM response ready")
        return resp
