config = dict(
    dataset_path="/data/robovqa/convert_data/val",
    split="val_my",
    processed_dataset_path="/code1/data/robobrain2-benchmark/robovqa/",
    processor="process.py",
)

post_prompt = """
"""

dataset = dict(
    type="VideoDataset",
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt=post_prompt,
    ),
    config=config,
    name="robovqa",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
