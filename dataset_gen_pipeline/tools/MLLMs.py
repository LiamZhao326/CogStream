import os
from openai import OpenAI
import base64
import requests
import json
import time
class GPT:
    def __init__(self, multi_turn=False, model="gpt-4o-2024-11-20", max_tokens=2048,temperature=0.8,
                 record=True):
        """
        Initialize the Qwen class with the OpenAI client and configuration parameters.

        Args:
            multi_turn (bool): Whether to maintain conversation history for multi-turn dialogues.
            model (str): The model to use for API calls (e.g., 'qwen-vl-max-2024-12-30').
            max_tokens (int): Maximum number of tokens for the response.
            temperature (float, optional): Sampling temperature for the model.
            record (bool): Whether to record and print token usage.
        """
        self.api_key=''
        self.api_url=''
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        self.conversation_history = []
        self.multi_turn = multi_turn
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.record = record
        self.total_tokens = 0

    def encode_image(self, image_path):
        """
        Encode an image file to a base64 string for vision tasks.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Base64 encoded string of the image.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def vision(self, prompt, imagesFetched=None, used_url=False):
        """
        Handle vision tasks by processing text prompts and optional images.

        Args:
            prompt (str): Text prompt for the vision task.
            imagesFetched (list, optional): List of image paths or URLs.
            used_url (bool): If True, treat imagesFetched as URLs; otherwise, encode as base64.

        Returns:
            str: The model's response, or None if the request fails after retries.
        """
        # Construct message content with text and optional images
        text_mes = {"type": "text", "text": prompt}
        if not imagesFetched:
            content = [text_mes]
        else:
            images_mes = []
            for img_dir in imagesFetched:
                if not used_url:
                    base64_image = self.encode_image(img_dir)
                    images_mes.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
                else:
                    images_mes.append({"type": "image_url", "image_url": {"url": img_dir}})
            content = [text_mes] + images_mes
        message = {"role": "user", "content": content}

        # Manage conversation history for multi-turn
        if self.multi_turn:
            self.conversation_history.append(message)

        messages_to_send = self.conversation_history if self.multi_turn else [message]

        # Attempt API call with retries
        retry = 0
        model = self.model
        while retry < 5:
            if retry == 3:
                time.sleep(2)  # Pause after 10 retries to avoid rate limits
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=messages_to_send,
                    model=model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                if chat_completion.choices is None: model = 'gpt-4o-2024-11-20'
                answer = chat_completion.choices[0].message.content
                # Record token usage if enabled
                if self.record:
                    self.total_tokens += chat_completion.usage.total_tokens
                    print(f"completion_tokens: {chat_completion.usage.completion_tokens}")
                    print(f"prompt_tokens: {chat_completion.usage.prompt_tokens}")
                    print(f"total_tokens: {self.total_tokens}")
                # Append assistant’s response to history if multi-turn
                if self.multi_turn:
                    self.conversation_history.append({"role": "assistant", "content": answer})
                return answer
            except Exception as e:
                print(f"Request failed: {e}")
                retry += 1
                continue
        raise ConnectionError("网络异常")

    def chat(self, prompt, system=None):
        """
        Handle text-based chat with an optional system message.

        Args:
            prompt (str): User’s text prompt.
            system (str, optional): System message to set the context.

        Returns:
            str: The model’s response, or None if the request fails after retries.
        """
        # Construct message with optional system prompt
        if system is not None:
            message = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        else:
            message = [{"role": "user", "content": prompt}]

        # Manage conversation history for multi-turn
        if self.multi_turn:
            self.conversation_history.append(message[0])  # Append system or user message

        messages_to_send = self.conversation_history if self.multi_turn else message

        # Attempt API call with retries
        retry = 0
        while retry < 20:
            if retry == 10:
                time.sleep(2)  # Pause after 10 retries to avoid rate limits
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=messages_to_send,
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                answer = chat_completion.choices[0].message.content
                # Record token usage if enabled
                if self.record:
                    self.total_tokens += chat_completion.usage.total_tokens
                    print(f"completion_tokens: {chat_completion.usage.completion_tokens}")
                    print(f"prompt_tokens: {chat_completion.usage.prompt_tokens}")
                    print(f"total_tokens: {self.total_tokens}")
                # Append assistant’s response to history if multi-turn
                if self.multi_turn:
                    self.conversation_history.append({"role": "assistant", "content": answer})
                return answer
            except Exception as e:
                print(f"Request failed: {e}")
                retry += 1
                continue
        raise ConnectionError("网络异常")

    def mix_chat(self, prompt, imagesFetched=None, used_url=True):

        content = []
        # 识别 '/image'
        parts = prompt.split('/image')
        for i, part in enumerate(parts):
            # 文本
            if part.strip():
                content.append({"type": "text", "text": part.strip()})
            # 图像 URL
            if imagesFetched and i < len(imagesFetched):
                content.append({"type": "image_url", "image_url": {"url": imagesFetched[i]}})
        # 构建完整消息
        message = [{"role": "user", "content": content}]

        # 添加用户输入到对话历史
        if self.multi_turn:
            self.conversation_history.append(message[0])

        # 准备请求数据
        # Manage conversation history for multi-turn
        if self.multi_turn:
            self.conversation_history.append(message)

        messages_to_send = self.conversation_history if self.multi_turn else [message]

        # Attempt API call with retries
        retry = 0
        while retry < 20:
            if retry == 10:
                time.sleep(2)  # Pause after 10 retries to avoid rate limits
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=messages_to_send,
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                answer = chat_completion.choices[0].message.content
                # Record token usage if enabled
                if self.record:
                    self.total_tokens += chat_completion.usage.total_tokens
                    print(f"completion_tokens: {chat_completion.usage.completion_tokens}")
                    print(f"prompt_tokens: {chat_completion.usage.prompt_tokens}")
                    print(f"total_tokens: {self.total_tokens}")
                # Append assistant’s response to history if multi-turn
                if self.multi_turn:
                    self.conversation_history.append({"role": "assistant", "content": answer})
                return answer
            except Exception as e:
                print(f"Request failed: {e}")
                retry += 1
                continue
        raise ConnectionError("网络异常")

