config = dict(
    dataset_path="/code1/data/robobrain2-benchmark/pixmo-points-eval",
    split="test",
    processed_dataset_path="/code1/data/robobrain2-benchmark/pixmo-points-eval",
    processor="process.py",
)

post_prompt1 = """Your task is to identify specific points in the image based on the question. Respond with a brief explanation if needed, followed by a list of 2D point coordinates.
Format your final answer strictly as follows on the last line of your response:
Answer: [(x1, y1), (x2, y2), ..., (xn, yn)]
Do not include additional text after this line.
"""

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt=post_prompt1,
    ),
    config=config,
    name="pixmo_points_eval",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate_robobrain.py")
