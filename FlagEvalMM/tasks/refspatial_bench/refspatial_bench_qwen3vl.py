config = dict(
    dataset_path="/code1/data/robobrain2-benchmark/RefSpatial-Bench",
    split="all",
    processed_dataset_path="/code1/data/robobrain2-benchmark/RefSpatial-Bench",
    processor="process.py",
)

post_prompt = """Output the point coordinates in JSON format.
For example: [
{"point_2d": [x, y], "label": "point_1"}
]
"""

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt=post_prompt,
    ),
    config=config,
    name="refspatial_bench",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate_real_qwen3vl.py")
