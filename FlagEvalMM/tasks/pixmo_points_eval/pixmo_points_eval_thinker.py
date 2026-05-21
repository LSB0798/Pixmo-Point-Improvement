config = dict(
    dataset_path="/data/lishuaibing/test_process/robobrain2-benchmark/pixmo-points-eval",
    split="test",
    processed_dataset_path="/data/lishuaibing/test_process/robobrain2-benchmark/pixmo-points-eval",
    processor="process.py",
)

post_prompt1 = """Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should range from 0 to 1000, representing the relative pixel positions of the points in the image."""

post_prompt2 = """Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...]. The coordinates should range from 0 to 1000."""

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt=post_prompt1,
    ),
    config=config,
    name="pixmo_points_eval",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
