config = dict(
    dataset_path="/code1/data/robobrain2-benchmark/RefSpatial-Bench",
    split="all",
    processed_dataset_path="/code1/data/robobrain2-benchmark/RefSpatial-Bench",
    processor="process.py",
)

post_prompt = """Your task is to identify specific points in the image based on the question. Respond with a brief explanation if needed, followed by a list of 2D point coordinates.
Format your final answer strictly as follows on the last line of your response:
Answer: [(x1, y1), (x2, y2), ..., (xn, yn)]
Do not include additional text after this line.
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

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
