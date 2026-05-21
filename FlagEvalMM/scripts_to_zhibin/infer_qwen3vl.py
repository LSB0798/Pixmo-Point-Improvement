
# /ogi-data/pzb-data/model/VLABench-main/weight/qwen3vl_1215
# /ogi-data/pzb-data/model/Qwen3-VL-4B-Instruct
# /code1/train_logs/merge_lora/thinker/v300-20251225-024907-alignment/iter_0001000

# CUDA_VISIBLE_DEVICES=1 vllm serve /ogi-data/pzb-data/model/Qwen3-VL-4B-Instruct \
#   --host 0.0.0.0 --port 8001 \
#   --served-model-name qwen3vl \
#   --limit-mm-per-prompt '{"image": 40, "video": 1}' \
#   --mm-processor-kwargs '{"fps": 2}' \
#   --allowed-local-media-path / \
#   --max-model-len 8192 \
#   --max-num-seqs 32 \
#   --gpu-memory-utilization 0.5

import argparse
import os
import time
from openai import OpenAI


def to_image_url(img: str) -> str:
    # 支持：http(s)://、file://、data:... 以及本地路径
    if img.startswith(("http://", "https://", "file://", "data:")):
        return img
    return "file://" + os.path.abspath(img)

# TEXT-cn = """你是一个专业的充电柜指示灯检测助手。你的任务是：根据用户提供的充电柜图片，识别黑色背板区域内的指示灯位置，以json格式输出所有坐标，例如：
# ```json
# [
#     {"bbox_2d": [x1, y1, x2, y2], "label": "充电柜指示灯"},
#     {"bbox_2d": [x1, y1, x2, y2], "label": "充电柜指示灯"}
# ]
# ```

# 【目标指示灯的定义】
# A) 位置：目标指示灯必须位于充电柜中间的黑色背板区域内。
# B) 数量与排列：目标最多只有两条，且上下排列，尺寸一致。
# C) 尺寸：目标灯条长度约 30 厘米；明显更长的灯带/氛围灯/装饰灯一律忽略。
# D) 相对关系（可选锚点）：若图片中能看到对应仓口开口/电池进出口，则目标指示灯位于其正上方。

# 【注意事项】
# A) 若候选多于两条：
# - 优先选择最符合“长度约 30cm 且两条尺寸最一致、上下成对”的那一对；
# - 忽略明显更长、更细、断续、偏离背板核心区域的光条。
# B) 若候选少于两条，指示灯可能被遮挡，被遮挡的指示灯对应列表为空，只返回可见的指示灯结果。

# 【最终输出格式】
# 框出每一个充电柜指示灯的位置，以json格式输出所有的坐标
# """

TEXT = """You are a professional charging-cabinet indicator light detection assistant. Your task is: based on a charging-cabinet image provided by the user, identify the indicator light(s) located within the black backplate area in the center of the cabinet, and output the coordinates of all detected lights in JSON format, for example:

```json
[
  {"bbox_2d": [x1, y1, x2, y2], "label": "Charging cabinet indicator light"},
  {"bbox_2d": [x1, y1, x2, y2], "label": "Charging cabinet indicator light"}
]
```

## Definition of the target indicator light

A) Location: The target indicator light must be inside the black backplate area in the middle of the charging cabinet.
B) Count & arrangement: There are at most two target light bars. If two exist, they are vertically stacked (one above the other) and equal in size.
C) Size: The target light bar is about 30 cm long. Ignore any obviously longer light strips / ambient lights / decorative lights.
D) Relative relationship (optional anchor): If the image shows the corresponding bay opening / battery inlet-outlet, the target indicator light is directly above it.

## Notes

A) If there are more than two candidates:

* Prefer the pair that best matches ~30 cm length and is most consistent in size, forming a vertical pair.
* Ignore light strips that are obviously longer, thinner, intermittent, or far from the core backplate area.

B) If there are fewer than two candidates, a light may be occluded. In that case, the output list should be empty for the occluded light—return only the visible indicator light(s).

## Final output format

Draw a bounding box around each charging-cabinet indicator light and output all coordinates in JSON format.
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://0.0.0.0:8000/v1")
    ap.add_argument("--model", default="qwen3vl")
    ap.add_argument("--image", required=True, help="图片路径 或 URL")
    ap.add_argument("--text", required=True, help="要问模型的文本")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()
    
    args.text = TEXT if args.text.strip() == "" else args.text

    client = OpenAI(api_key="EMPTY", base_url=args.base_url, timeout=args.timeout)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": to_image_url(args.image)}},
                {"type": "text", "text": args.text},
            ],
        }
    ]

    start = time.time()
    resp = client.chat.completions.create(
        model=args.model,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(f"Latency: {time.time() - start:.2f}s")
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()

# python light_sh_yjp/infer_qwen3vl.py \
#   --base-url http://0.0.0.0:8000/v1 \
#   --image ./light_data/train/off_unkown/20251224-102019/frame_001110.jpg \
#   --text "" \
#   --temperature 0.0


# Please provide the bounding box coordinate of the region this sentence describes: charging-cabinet indicator light. The bbox coordinate format is [top-left x, top-left y, bottom-right x, bottom-right y].All values must be integer points between 0 and 1000, inclusive.