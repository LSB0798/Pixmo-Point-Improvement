config = dict(
    dataset_path="/code1/data/robobrain2-benchmark/RefSpatial-Bench",
    split="all",
    processed_dataset_path="/code1/data/robobrain2-benchmark/RefSpatial-Bench",
    processor="process.py",
)

post_prompt = """Please provide its 2D coordinates. Your answer should be formatted as a tuple, i.e. [(x, y)], where the tuple contains the x and y coordinates of a point satisfying the conditions above."""

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt=post_prompt,
    ),
    config=config,
    name="refspatial_bench",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate_qwen3vl.py")