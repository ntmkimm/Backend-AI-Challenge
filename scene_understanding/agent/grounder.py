import os
from typing import Dict, Optional, List
import openai
from PIL import Image
import re
# It is assumed that TStar.utilites defines the following functions:
# - encode_image_to_base64: converts a PIL.Image to a base64 string.
# - load_video_frames: loads a specified number of frames from a video.
import time

import math
import base64
import io
import os
from typing import List
import numpy as np
from PIL import Image
import cv2


def encode_image_to_base64(image) -> str:
    """
    Convert an image (PIL.Image or numpy.ndarray) to a Base64 encoded string.
    
    Args:
        image: A PIL.Image or numpy.ndarray representing the image.
    
    Returns:
        A Base64 encoded string of the image.
    
    Raises:
        ValueError: If the input is neither a PIL.Image nor a numpy.ndarray.
    """
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL.Image or numpy.ndarray")
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        raise ValueError(f"Error encoding image: {str(e)}")


def load_video_frames(video_path: str, num_frames: int = 8) -> List[Image.Image]:
    """
    Load a specified number of frames from a video as PIL.Image objects.
    
    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to extract.
    
    Returns:
        A list of PIL.Image objects representing the extracted frames.
    
    Raises:
        ImportError: If OpenCV is not installed.
        ValueError: If the video cannot be opened or has zero frames.
    """
    if cv2 is None:
        raise ImportError("OpenCV is not installed, cannot load video frames.")
    
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        raise ValueError("Video has zero frames or could not retrieve frame count.")
    
    num_frames = min(num_frames, total_frames)
    step = total_frames / num_frames

    for i in range(num_frames):
        frame_index = int(math.floor(i * step))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    return frames


class GPT4Interface:
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Initialize the GPT-4 API client. The API key is read from the environment
        variable OPENAI_API_KEY if not provided.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model
        if not self.api_key:
            raise ValueError("Environment variable OPENAI_API_KEY is not set.")
        openai.api_key = self.api_key

    def _build_messages(self, system_message: str, user_content: List) -> List[Dict]:
        """
        Build the messages list required by the OpenAI API.
        """
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]

    def _encode_frames(self, frames: List[Image.Image]) -> List[Dict]:
        """
        Encode image frames into Base64 formatted messages.
        """
        messages = []
        for i, frame in enumerate(frames):
            try:
                frame_base64 = encode_image_to_base64(frame)
                visual_context = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_base64}",
                        "detail": "low"
                    }
                }
                messages.append(visual_context)
            except Exception as e:
                raise ValueError(f"Error encoding frame {i}: {str(e)}")
        return messages

    def inference_text_only(
        self,
        query: str,
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Perform inference using GPT-4 API for text-only input.
        """
        messages = self._build_messages(system_message, query)
        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"

    def _inference_with_frames(
        self,
        query: str,
        frames: List[Image.Image],
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Perform inference using GPT-4 API with frames as context.
        """
        user_content = [{"type": "text", "text": query}]
        try:
            user_content.extend(self._encode_frames(frames))
        except ValueError as e:
            return str(e)
        messages = self._build_messages(system_message, user_content)
        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"

    def inference_qa(
        self,
        question: str,
        options: str,
        frames: Optional[List[Image.Image]] = None,
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        Perform multiple-choice inference using GPT-4 API.
        
        Args:
            question: The question to answer.
            options: Multiple-choice options as a string.
            frames: Optional visual context.
        
        Returns:
            The selected option (e.g., A, B, C, D).
        """
        query = (
            f"Question: {question}\nOptions: {options}\n"
            "Answer with the letter corresponding to the best choice."
        )
        user_content = [{"type": "text", "text": query}]
        if frames:
            try:
                user_content.extend(self._encode_frames(frames))
            except ValueError as e:
                return str(e)
        messages = self._build_messages(system_message, user_content)
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"

    def inference_with_frames(
        self,
        query: str,
        frames: List[Image.Image],
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        A unified inference interface supporting mixed text and image inputs.
        The query may include <image> tags.
        """
        parts = query.split("<image>")
        user_content = []
        for i, part in enumerate(parts):
            if part.strip():
                user_content.append({"type": "text", "text": part.strip()})
            if i < len(frames):
                try:
                    frame_base64 = encode_image_to_base64(frames[i])
                    visual_context = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame_base64}",
                            "detail": "low"
                        }
                    }
                    user_content.append(visual_context)
                except Exception as e:
                    return f"Error encoding frame {i}: {str(e)}"
        messages = self._build_messages(system_message, user_content)
        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"


class TStarUniversalGrounder:
    """
    Combines functionalities of TStarGrounder and TStarGPTGrounder.
    Allows switching between LlavaInterface and GPT4Interface via the backend parameter.
    """
    def __init__(
        self,
        model_name: str = "gpt-4o",
        model_path: Optional[str] = None,
        model_base: Optional[str] = None,
        gpt4_api_key: Optional[str] = None,
        num_frames: Optional[int] = 8,
    ):
        self.backend = model_name.lower()
        self.num_frames = num_frames
        # if "llava" in self.backend:
        #     if not model_path:
        #         raise ValueError("Please provide model_path for LlavaInterface")
        #     self.VLM_model_interface = LlavaInterface(model_path=model_path, model_base=model_base)
        # elif "qwen" in self.backend:
        #     # Initialize QwenInterface if 'qwen' is specified in the backend.
        #     self.VLM_model_interface = QwenInterface(model_name=model_name, device="auto")
        if "gpt" in self.backend:
            self.VLM_model_interface = GPT4Interface(model=model_name, api_key=gpt4_api_key)
        else:
            raise ValueError("backend must be one of: 'llava', 'qwen', or 'gpt4'.")

    def inference_query_grounding(
        self,
        video_path: str,
        question: str,
        temperature: float = 0.0,
        max_tokens: int = 512
    ) -> Dict[str, List[str]]:
        """
        Identify target objects and cue objects from the video based on the question.
        
        Args:
            video_path: Path to the video file.
            question: The question.
            options: (Optional) multiple-choice options.
        
        Returns:
            A dictionary with two keys: target_objects and cue_objects.
        """
        frames = load_video_frames(video_path=video_path, num_frames=self.num_frames)
        system_prompt = (
            "Here is a video:\n" + "\n".join(["<image>"] * len(frames)) +
            "\nHere is a question about the video:\n" +
            f"Question: {question}\n"
        )
        # system_prompt += f"Options: {options}\n"
        system_prompt += (
            "\nWhen answering this question about the video:\n"
            # "1. Identify key objects that can locate the answer (list key objects, separated by commas).\n"
            "1. Identify only one key objects that can locate the answer.\n Example Answer: key objects: car"
            # "2. Identify cue objects that might be near the key objects and appear in the scenes (list cue objects, separated by commas).\n\n"
            # "Provide your answer in two lines, listing the key objects and cue objects separated by commas."
            # "Provide your answer in one lines, listing the key objects separated by commas, with their importance sorted from high to low."
        )
        response = self.VLM_model_interface.inference_with_frames(
            query=system_prompt,
            frames=frames,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        if len(lines) != 1:
            raise ValueError(f"Unexpected response format --> {response}")

        target_objects = [self.check_objects_str(obj) for obj in lines[0].split(",") if obj.strip()]
        # cue_objects = [self.check_objects_str(obj) for obj in lines[1].split(",") if obj.strip()]
        cue_objects = []
        return target_objects, cue_objects

    def check_objects_str(self, obj: str) -> str:
        """
        Process the object string to normalize object names by:
        - Lowercasing
        - Removing prefixes like "1. ", "2. ", "Key objects:"
        - Removing punctuation
        - Stripping extra whitespace
        """
        obj = obj.strip().lower()

        # Remove known prefixes (with optional whitespace)
        obj = re.sub(r"^(key objects|cue objects)?[:\-]?\s*", "", obj)
        obj = obj.replace("key objects: ", "").replace(": ", "")
        obj = re.sub(r"^[0-9]+\.\s*", "", obj)  # e.g., "1. "
        
        # Remove punctuation like periods, colons etc.
        obj = re.sub(r"[^\w\s-]", "", obj)  # Keep letters, numbers, space, hyphen

        return obj.strip()

    def inference_qa(
        self,
        frames: List[Image.Image],
        question: str,
        temperature: float = 0.2,
        max_tokens: int = 128
    ) -> str:
        """
        Perform multiple-choice inference and return the most likely option (e.g., A, B, C, D).
        """
        system_prompt = (
            "Thinking carefully and answer the question below\n" +
            "\n".join(["<image>"] * len(frames)) +
            f"\nQuestion: {question}\n" +
            "Answer directly."
        )
        response = self.VLM_model_interface.inference_with_frames(
            query=system_prompt,
            frames=frames,
            temperature=temperature,
            max_tokens=30
        )
        return response.strip()

    def inference_openend_qa(
        self,
        frames: List[Image.Image],
        question: str,
        temperature: float = 0.2,
        max_tokens: int = 2048
    ) -> str:
        """
        Perform open-ended question answering based on the video.
        """
        system_prompt = (
            "Answer the following question briefly based on the video.\n" +
            "\n".join(["<image>"] * len(frames)) +
            f"\nQuestion: {question}\n"
        )
        response = self.VLM_model_interface.inference_with_frames(
            query=system_prompt,
            frames=frames,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.strip()


if __name__ == "__main__":
    # Test example.
    start_time = time.time()
    video_path = "/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/video_shot/L01_V001_shots/L01_V001_shot_0221_022394-022509.mp4"
    frames = load_video_frames(video_path=video_path, num_frames=2)
    print("\n=== Using GPT-4 backend ===")
    gpt4_grounder = TStarUniversalGrounder(
        model_name="gpt-4o",
        gpt4_api_key=os.getenv("OPENAI_API_KEY"),
        num_frames=2
    )
    # searchable_objects = gpt4_grounder.inference_query_grounding(
    #     video_path=video_path,
    #     question="there is a man hold cheese, how many part of cheese that he get?"
    # )
    # print("GPT-4 Grounding Result:", searchable_objects)

    question_mc = "there is a man hold cheese, how many part of cheese that he get?\n"
    answer_gpt4 = gpt4_grounder.inference_qa(frames, question_mc)
    print("GPT-4 QA Answer:", answer_gpt4)
    end_time = time.time()
    print("time inference: ", end_time - start_time)