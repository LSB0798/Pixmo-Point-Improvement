import shutil
from pathlib import Path

base_dir = Path("light_results_yjp/easyprompt2-outputpoint*2-en-thinker")

vis_dir = base_dir / "vis"
preds_dir = base_dir / "preds"

src_dirs = [vis_dir / "true", vis_dir / "false"]

dst_vis_dir = base_dir / "sample_200" / "vis"
dst_preds_dir = base_dir / "sample_200" / "preds"
dst_vis_dir.mkdir(parents=True, exist_ok=True)
dst_preds_dir.mkdir(parents=True, exist_ok=True)

exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def unique_dst_path(dst_dir: Path, filename: str) -> Path:
    out = dst_dir / filename
    if not out.exists():
        return out
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    k = 1
    while True:
        cand = dst_dir / f"{stem}_{k}{suffix}"
        if not cand.exists():
            return cand
        k += 1

# 1) 收集图片
images = []
for d in src_dirs:
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(f"源目录不存在或不是文件夹: {d}")
    images.extend([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts])

images = sorted(images, key=lambda p: (p.name, str(p)))
n = len(images)
assert n >= 200, f"图片数量不足，只有 {n} 张"

# 2) 等间隔抽取 200 张
indices = [int(i * n / 200) for i in range(200)]
selected = [images[i] for i in indices]

# 3) 复制图片 + 对应 preds json
missing_json = []
copied = 0

for img_path in selected:
    # copy image
    dst_img_path = unique_dst_path(dst_vis_dir, img_path.name)
    shutil.copy2(img_path, dst_img_path)

    # find & copy json with same stem
    json_name = img_path.with_suffix(".json").name.replace("_vis", "")
    json_path = preds_dir / json_name

    if json_path.exists():
        # 保持和图片一致的命名策略：如果图片因重名被加 _1，那么 json 也跟着用同名（stem一致）
        # 例如 dst_img_path: xxx_1.png -> json 用 xxx_1.json（优先），否则退回原始 xxx.json
        desired_json_name = dst_img_path.with_suffix(".json").name
        desired_json_path = preds_dir / desired_json_name
        src_json_path = desired_json_path if desired_json_path.exists() else json_path

        dst_json_path = unique_dst_path(dst_preds_dir, dst_img_path.with_suffix(".json").name)
        shutil.copy2(src_json_path, dst_json_path)
    else:
        missing_json.append(json_path)

    copied += 1

print(f"已从 {len(src_dirs)} 个目录汇总 {n} 张图片，等间隔抽取 {copied} 张。")
print(f"图片保存至: {dst_vis_dir}")
print(f"JSON 保存至: {dst_preds_dir}")
if missing_json:
    print(f"⚠️ 有 {len(missing_json)} 个图片未找到对应 JSON（示例前5个）：")
    for p in missing_json[:5]:
        print("  ", p)
