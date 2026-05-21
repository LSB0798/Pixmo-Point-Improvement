config = dict(
    dataset_path="/code1/data/robobrain2-benchmark/moving_box",
    split="val",
    processed_dataset_path="/code1/data/robobrain2-benchmark/moving_box",
    processor="process.py",
)

post_prompt = """"""

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt=post_prompt,
    ),
    config=config,
    name="moving_box_val",
)

evaluator = dict(
    type="BaseEvaluator",
    eval_func="evaluate.py",
    detailed_keys=["sub_task"],
)

