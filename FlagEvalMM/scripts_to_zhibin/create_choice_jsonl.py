import json
import os
from pathlib import Path

def generate_charge_cabinet_jsonl(
    root_img_dir: str,
    output_jsonl_path: str,
    default_upper_status: str = "green",  # 默认上指示灯状态
    default_lower_status: str = "green",  # 默认下指示灯状态
    manual_annotation: bool = False  # 是否手动标注（True=手动输入，False=用默认值）
):
    """
    遍历图片目录，生成充电柜指示灯状态的JSONL文件
    
    Args:
        img_dir: 图片所在目录
        output_jsonl_path: 输出JSONL文件路径
        default_upper_status: 上指示灯默认状态（手动标注时无效）
        default_lower_status: 下指示灯默认状态（手动标注时无效）
        manual_annotation: 是否手动输入每个图片的指示灯状态
    """
    # 校验状态合法性
    valid_status = {"green", "yellow", "red", "off", "unknown"}
    if default_upper_status not in valid_status:
        raise ValueError(f"默认上指示灯状态不合法，可选值：{valid_status}")
    if default_lower_status not in valid_status:
        raise ValueError(f"默认下指示灯状态不合法，可选值：{valid_status}")

    # 获取目录下的所有图片（支持常见格式）
    img_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    img_paths = []
    for root, dirs, files in os.walk(root_img_dir):
        for file in files:
            if file.lower().endswith(img_extensions):
                # 拼接完整路径
                abs_path = os.path.abspath(os.path.join(root, file))
                img_path = abs_path
                # 可选：转为相对路径（相对于根目录）
                # if use_relative_path:
                #     img_path = os.path.relpath(abs_path, root_img_dir)
                    
                img_paths.append(img_path)

    if not img_paths:
        raise FileNotFoundError(f"图片根目录 {root_img_dir} 下无有效图片")
    print(f"共发现 {len(img_paths)} 张图片（含子目录）")

    # 生成JSONL内容
    jsonl_lines = []
    user_prompt = """The indicator light in the image is marked with a blue circle. Please identify the status (color) of this indicator light. Choose from the following options:
A. Green
B. Yellow
C. Red
D. off (indicator light is off)
E. unknown (blocked or hard to distinguish)
Answer with the option's letter from the given choices directly."""

    for img_path in img_paths:
        # 手动标注指示灯状态
        if manual_annotation:
            print(f"\n处理图片：{img_path}")
            while True:
                upper = input("请输入上指示灯状态（green/yellow/red/off/unkown）：").strip().lower()
                if upper in valid_status:
                    break
                print(f"输入无效，请选择：{valid_status}")
            while True:
                lower = input("请输入下指示灯状态（green/yellow/red/off/unkown）：").strip().lower()
                if lower in valid_status:
                    break
                print(f"输入无效，请选择：{valid_status}")
        else:
            # 使用默认状态
            upper = default_upper_status
            lower = default_lower_status

        # 构造2条JSON数据
        data = {
            "messages": [
                # {"role": "system", "content": system_prompt},
                {"role": "user", "content": "<image>" + user_prompt},
                {"role": "assistant", "content": json.dumps({
                    "upper_indicator_light": upper,
                }, ensure_ascii=False)}
            ],
            "images": [img_path],
            "light_position": "upper"
        }
        # 转为JSON字符串并添加到列表
        jsonl_lines.append(json.dumps(data, ensure_ascii=False))
        data = {
            "messages": [
                # {"role": "system", "content": system_prompt},
                {"role": "user", "content": "<image>" + user_prompt},
                {"role": "assistant", "content": json.dumps({
                    "lower_indicator_light": lower
                }, ensure_ascii=False)}
            ],
            "images": [img_path],
            "light_position": "lower"
        }
        # 转为JSON字符串并添加到列表
        jsonl_lines.append(json.dumps(data, ensure_ascii=False))

    # 写入JSONL文件
    Path(os.path.dirname(output_jsonl_path)).mkdir(parents=True, exist_ok=True)
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(jsonl_lines))

    print(f"\n生成完成！JSONL文件已保存到：{output_jsonl_path}")
    print(f"共处理 {len(img_paths)} 张图片")


# ==================== 运行示例 ====================
if __name__ == "__main__":
    # 配置参数
    IMAGE_DIR = "/media/ubtrobot/b75d4911-7092-47c8-94b3-739fedb14a90/code/light_/lightVideo/go_image_val"  # 图片所在目录（替换为你的目录）
    OUTPUT_JSONL = "./green_yellow_val.jsonl"  # 输出JSONL路径
    USE_MANUAL_ANNOTATION = False  # True=手动标注每个图片的状态，False=用默认值

    generate_charge_cabinet_jsonl(
        root_img_dir=IMAGE_DIR,
        output_jsonl_path=OUTPUT_JSONL,
        default_upper_status="green",
        default_lower_status="yellow",
        manual_annotation=USE_MANUAL_ANNOTATION
    )