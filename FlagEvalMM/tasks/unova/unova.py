# unova.py
# UNOVA 拆垛评测任务配置

config = dict(
    # 这里可以是：
    #   1) 一个 jsonl 文件路径（字符串）
    #   2) 多个 jsonl 的列表
    #   3) 存放 jsonl 的目录（process.py 里会做处理）
    dataset_path=[
        "/code1/data/unova/test.jsonl",   # TODO: 换成你自己的 GT JSONL 路径
    ],
    split="test",
    processed_dataset_path="/code1/data/unova/flag_eval/",  # 输出 data.json 的位置
    processor="process.py",
)

# 直接把 eval_unova.py 里的 SYSTEM_PROMPT + USER_TEXT 拼成 post_prompt
post_prompt = (
    "你是抓取规划器。任务：从「托盘（栈板）上的箱子堆垛」中选择并规划抓取；"
    "若距离堆垛过远先给出靠近点，若已接近则给出抓取目标与站位。 "
    "对象与范围：箱子：塑料/周转/物流收纳箱等（颜色不限）。仅考虑「位于托盘上的堆垛」的箱子；"
    "忽略地面/桌面/货架等非托盘上的箱子。 "
    "选箱规则：1) 最上层优先。2) 若同层，选最右（以 bbox 中心 x 最大为准）。 "
    "站位规则：目标左侧有同层邻箱 → \"right\"；目标右侧有同层邻箱 → \"left\"；两侧都无 → \"middle\"。 "
    "远/近与靠近点：无法可靠给出目标框时视为「远」，返回堆垛正面约 1m 的落脚点投影（图像像素点）。 "
    "输出格式（只输出其一）：远："
    "{\"state\":\"approach\",\"approach_point_2d\":[x,y]} "
    "近：{\"state\":\"select\",\"target_bbox_2d\":[x1,y1,x2,y2],"
    "\"stance_flag\":\"left|middle|right\"} "
    "约束：坐标为像素整型，bbox 为左上-右下且在图像范围内。只返回一个 JSON，不要任何其他文字。"
    "\n\n"
    "现在请根据图像输出当前的最优拆垛信息。"
)

dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(
        type="PromptTemplate",
        # VqaBaseDataset 通常会做：final_prompt = question + post_prompt
        # 这里我们把完整指令都放在 post_prompt 里，question 留空即可。
        post_prompt=post_prompt,
    ),
    name="unova",
)

# 评测逻辑放在同目录的 evaluate.py 里
evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
