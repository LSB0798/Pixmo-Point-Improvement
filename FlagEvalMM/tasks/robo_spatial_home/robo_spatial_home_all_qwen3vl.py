config = dict(
    dataset_path="/code1/data/robobrain2-benchmark/RoboSpatial-Home",
    split="all",
    processed_dataset_path="/code1/data/robobrain2-benchmark/RoboSpatial-Home",
    processor="process.py",
)

post_prompt_point = """Output the point coordinates in JSON format.
For example: [
{"point_2d": [x, y], "label": "point_1"}
]
"""

post_prompt_yes_no = """Your task is to answer the question above. \n\nFormat your final answer strictly as follows: yes or no\n\nDo not include additional text after this line."""


def post_prompt(question_type: str, **kwargs):
    if question_type == "point":
        return post_prompt_point
    else:
        return post_prompt_yes_no


dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt=post_prompt,
    ),
    config=config,
    name="robo_spatial_home_all",
)

evaluator = dict(type="BaseEvaluator", eval_func="robo_spatial_evaluate_real_qwen3vl.py")


