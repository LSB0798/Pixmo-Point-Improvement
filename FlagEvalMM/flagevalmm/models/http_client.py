import requests.models
import json
import requests
import httpx
import re
import os
from pathlib import Path
from typing import Optional, List, Any, Union, Dict
from flagevalmm.common.logger import get_logger
from flagevalmm.models.base_api_model import BaseApiModel
from flagevalmm.models.api_response import ApiResponse, ApiUsage
from flagevalmm.prompt.prompt_tools import encode_image
from flagevalmm.common.video_utils import load_image_or_video
from PIL import Image

logger = get_logger(__name__)

IMAGE_REGEX = r"<image \d+>"

def _is_image(path: str) -> bool:
    return os.path.splitext(path.lower())[1] in {
        ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"
    }

def _is_video(path: str) -> bool:
    return os.path.splitext(path.lower())[1] in {
        ".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".m4v"
    }

class HttpClient(BaseApiModel):
    def __init__(
        self,
        model_name: str,
        chat_name: Optional[str] = None,
        max_tokens: int = 32768,
        temperature: float = 0.0,
        max_image_size: Optional[int] = None,
        min_short_side: Optional[int] = None,
        max_long_side: Optional[int] = None,
        max_num_frames: Optional[int] = 16,
        use_cache: bool = False,
        api_key: Optional[str] = None,
        url: Optional[Union[str, httpx.URL]] = None,
        reasoning: Optional[Dict[str, Any]] = None,
        provider: Optional[Dict[str, Any]] = None,
        retry_time: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=model_name,
            chat_name=chat_name,
            max_tokens=max_tokens,
            temperature=temperature,
            max_image_size=max_image_size,
            min_short_side=min_short_side,
            max_long_side=max_long_side,
            max_num_frames=max_num_frames,
            use_cache=use_cache,
            reasoning=reasoning,
            provider=provider,
            retry_time=retry_time,
            **kwargs,
        )
        self.url = url
        self.headers = {"Content-Type": "application/json"}
        if self.url and "azure.com" in self.url.lower():
            self.headers["api-key"] = api_key
        else:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _chat(self, chat_messages: Any, **kwargs):
        chat_args = self.chat_args.copy()
        chat_args.update(kwargs)
        data = {"model": f"{self.model_name}", "messages": chat_messages, **chat_args}
        # 观测：打印最后一条 user 的 content 类型序列与 payload 尺寸
        # try:
        #     parts = [p.get("type") for p in chat_messages[-1].get("content", [])]
        #     size = len(json.dumps(data))
        #     logger.info(f"[payload parts] {parts} | bytes={size}")
        # except Exception:
        #     pass
        if self.stream is False:
            yield from self._non_streaming_chat(data)
        else:
            yield from self._streaming_chat(data)

    def _non_streaming_chat(self, data):
        """Handle non-streaming API requests."""
        if not hasattr(self, "retry_time"):
            retry_time = 300
        else:
            retry_time = self.retry_time

        response = requests.post(
            self.url,
            headers=self.headers,
            data=json.dumps(data),
            timeout=retry_time,
        )
        try:
            response_json = response.json()
        except Exception as e:
            raise Exception(f"Error: {response.text}, {e}")
        if response.status_code != 200:
            if "error" not in response_json:
                yield ApiResponse.from_content(
                    f"Error code: {response_json['message']}"
                )
                return
            err_msg = response_json["error"]
            if "code" in err_msg and "message" in err_msg:
                if (
                    err_msg["code"] == "data_inspection_failed"
                    or err_msg["code"] == "1301"
                    or "no candidates" in err_msg["message"].lower()
                ):
                    yield ApiResponse.from_content(err_msg["message"])
                    return
            raise Exception(
                f"Request failed with status code {response.status_code}: {err_msg}"
            )

        # Parse usage information if available
        usage = None
        if "usage" in response_json:
            usage = ApiUsage.from_dict(response_json["usage"])

        if "choices" in response_json:
            message = response_json["choices"][0]["message"]
            res = ""
            reasoning_content = message.get(
                "reasoning_content", message.get("reasoning", "")
            )
            if reasoning_content:
                res = f"<think>{reasoning_content}</think>\n"
            if "content" in message:
                res += message["content"]
                yield ApiResponse(content=res, usage=usage)
            else:
                yield ApiResponse(content="", usage=usage)
        else:
            yield ApiResponse(
                content=response_json["completions"][0]["text"], usage=usage
            )

    def _streaming_chat(self, data):
        """Handle streaming API requests."""
        think_start = False
        with requests.post(
            self.url,
            headers=self.headers,
            data=json.dumps(data),
            stream=True,
            timeout=300,
        ) as response:
            if response.status_code != 200:
                raise Exception(
                    f"Stream request failed with status code {response.status_code}: {response.text}"
                )

            for line in response.iter_lines():
                if line:
                    # Remove "data: " prefix if it exists (common in SSE)
                    line_text = line.decode("utf-8")
                    if '"usage":null' not in line_text:
                        print(f"line_text: {line_text}")
                    if line_text.startswith("data: "):
                        line_text = line_text[6:]

                    # Skip heartbeat or empty messages
                    if line_text.strip() == "" or line_text == "[DONE]":
                        continue

                    try:
                        chunk = json.loads(line_text)
                        # Extract content from the chunk based on API response format
                        if "choices" in chunk:
                            delta = chunk["choices"][0].get("delta", {})
                            if (
                                "reasoning_content" in delta
                                and delta["reasoning_content"]
                            ):
                                content = delta["reasoning_content"]
                                if think_start is False:
                                    think_start = True
                                    content = f"<think>{content}"
                                yield ApiResponse.from_content(content)
                            if "content" in delta and delta["content"]:
                                content = delta["content"]
                                if think_start:
                                    content = f"</think>\n{content}"
                                    think_start = False
                                if chunk.get("usage") is not None:
                                    usage = ApiUsage.from_dict(chunk["usage"])
                                    yield ApiResponse(content=content, usage=usage)
                                else:
                                    yield ApiResponse.from_content(content)
                    except json.JSONDecodeError as e:
                        raise Exception(
                            f"Failed to parse chunk: {line_text}, error: {e}"
                        )

    # ---------------- Qwen3-VL-MoE 对齐：视频按“视频”发送，而非抽帧成图 ----------------
    def _to_file_url(self, p: str) -> str:
        """将本地路径转成标准 file:///... URI；若已是 http(s)/file 直接返回。"""
        if isinstance(p, str) and (p.startswith("http://") or p.startswith("https://") or p.startswith("file://")):
            return p
        return Path(p).resolve().as_uri()  # 标准三斜杠

    def _append_video_block(self, content_list, video_path: str):
        """
        构造一个“视频”内容块（不抽帧），对齐 qwen3-vl-moe：
        {
          "type": "video_url",
          "video_url": {"url": "file:///.../xxx.mp4"},
        }
        """
        content_list.append(
        {
            "type": "video_url",
            "video_url": {"url": self._to_file_url(video_path)}
        }
    )

    def _append_image_block(self, content_list, image_path: str):
        """
        构造一个“图像”内容块，对齐 qwen3-vl-moe：
        {
          "type": "image_url",
          "image_url": {"url": "file:///.../xxx.jpg"}
        }
        """
        # base64_image = encode_image(
        #     image_path,
        #     max_size=self.max_image_size,
        #     min_short_side=self.min_short_side,
        #     max_long_side=self.max_long_side,
        # )
        content_list.append(
            {
                "type": "image_url",
                # "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                "image_url": {"url": self._to_file_url(image_path)},
            }
        )

    def build_message(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        multi_modal_data: Dict[str, Any] = {},
        past_messages: Optional[List] = None,
    ) -> List:
        """
        严格对齐 qwen3-vl-moe 的多模态消息规则：
          1) 只有图像或只有视频：先媒体后文本
          2) 视频  最后一张图像（ending frame）：video -> image -> ". "  query
          3) 其它混合：按输入顺序加入媒体，最后加文本
        兼容：若无视频但 query 含 <image i>，走原有 interleave 逻辑。
        """
        messages = list(past_messages) if past_messages else []
        system_prompt = system_prompt if system_prompt else self.system_prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 统一收集图像/视频；允许单个或列表
        images_path, videos_path = [], []
        if isinstance(multi_modal_data, dict):
            for k in ("image", "images"):
                if k in multi_modal_data and multi_modal_data[k]:
                    v = multi_modal_data[k]
                    images_path.extend(v if isinstance(v, list) else [v])
            for k in ("video", "videos"):
                if k in multi_modal_data and multi_modal_data[k]:
                    v = multi_modal_data[k]
                    videos_path.extend(v if isinstance(v, list) else [v])
        
        images = [p for p in images_path + videos_path if _is_image(p)]
        videos = [p for p in videos_path + images_path if _is_video(p)]

        # 如果无视频但 query 有 <image i> 引用，则使用原有的插图规则（保持兼容）
        # 删除<image>
        query = re.sub('<image>', '', query)
        if len(videos) == 0 and re.search(IMAGE_REGEX, query or "") and len(images) > 0:
            self.build_interleaved_message(query, messages, images)
            return messages

        # 单条 user 消息，精确控制媒体顺序
        content = []
        has_images = len(images) > 0
        has_videos = len(videos) > 0

        if has_images and not has_videos:
            # 情况 1：只有图像 → 先图像后文本
            for img in images:
                self._append_image_block(content, img)
            content.append({"type": "text", "text": query})

        elif has_videos and not has_images:
            # 情况 2：只有视频 → 先视频后文本（不抽帧）
            for vid in videos:
                self._append_video_block(content, vid)
            content.append({"type": "text", "text": query})

        elif has_videos and has_images:
            # 情况 3：视频  最后一张图像（ending frame）
            # 触发条件：有视频且至少 1 张图；若只有 1 张图，认为它是 ending frame（与 v3_moe 一致）
            if len(images) == 1:
                content.append({"type": "text", "text": "According to video "})
                for vid in videos:
                    self._append_video_block(content, vid)
                content.append({"type": "text", "text": " and ending frame "})
                self._append_image_block(content, images[-1])
                # content.append({"type": "text", "text": "According to video and ending frame." + query})
                content.append({"type": "text", "text": query})
            else:
                # 情况 4：其它混合 → 媒体按输入顺序加入，再文本
                # 若 multi_modal_data 里有你们自定义的顺序（例如 "ordered"），可在此优先使用
                # 这里简单按 videos 再 images（或你可以按你们真实的输入序列来排）
                for vid in videos:
                    self._append_video_block(content, vid)
                for img in images:
                    self._append_image_block(content, img)
                content.append({"type": "text", "text": query})
        else:
            # 无多模态 → 纯文本
            content.append({"type": "text", "text": query})

        # import pdb; pdb.set_trace()
        messages.append({"role": "user", "content": content})
        return messages

    def build_interleaved_message(
        self, query: str, messages: List, image_data: List[Union[str, Image.Image]]
    ):
        referenced_numbers = [
            int(re.search(r"\d+", ref).group())
            for ref in re.findall(IMAGE_REGEX, query)
        ]
        content = []
        # Check if all referenced numbers are valid
        if referenced_numbers:
            max_ref = max(referenced_numbers)
            min_ref = min(referenced_numbers)
            if max_ref > len(image_data) or min_ref < 1:
                raise ValueError("Invalid image reference in question.")

        base64_images = [
            encode_image(
                data,
                max_size=self.max_image_size,
                min_short_side=self.min_short_side,
                max_long_side=self.max_long_side,
            )
            for data in image_data
        ]

        parts = re.split(r"(<image \d+>)", query)
        for part in parts:
            if len(part.strip()) == 0:
                continue
            if re.match(IMAGE_REGEX, part):
                # It's an image reference
                num = int(re.search(r"\d+", part).group())
                base64_image = base64_images[num - 1]
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                )
            else:
                assert len(part.strip()) > 0, f"part: {part}"
                # It's a text part
                content.append({"type": "text", "text": part})
        # If there are no referenced images, add all images to the message, not interleaved
        if not referenced_numbers:
            for base64_image in base64_images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                )
        messages.append({"role": "user", "content": content})
