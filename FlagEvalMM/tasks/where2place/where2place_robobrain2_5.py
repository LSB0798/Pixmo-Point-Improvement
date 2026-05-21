config = dict(
    dataset_path="/code1/data/robobrain2-benchmark/Where2Place",
    split="test",
    processed_dataset_path="/code1/data/robobrain2-benchmark/Where2Place/",
    processor="process.py",
)

post_prompt1 = """Please provide its 2D coordinates. Your answer should be formatted as a tuple, i.e. [(x, y)], where the tuple contains the x and y coordinates of a point satisfying the conditions above."""

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt=post_prompt1,
    ),
    config=config,
    name="Where2Place",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate_qwen3vl.py")