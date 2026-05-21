config = dict(
    dataset_path="/data/lishuaibing/test_process/robobrain2-benchmark/Where2Place",
    split="test",
    processed_dataset_path="/data/lishuaibing/test_process/robobrain2-benchmark/Where2Place/",
    processor="process.py",
)

post_prompt1 = """Your task is to identify specific points in the image based on the question. Respond with a brief explanation if needed, followed by a list of 2D point coordinates.
Each point should be represented as a normalized (x, y) tuple, where both x and y values are floats between 0 and 1, corresponding to the position within the image (e.g., for a point at pixel (50, 75) in a 100*100 image, the normalized coordinate is (0.5, 0.75)).
Format your final answer strictly as follows on the last line of your response:
Answer: [(x1, y1), (x2, y2), ..., (xn, yn)]
Do not include additional text after this line.
"""

post_prompt2 = """Your task is to identify specific points in the image based on the question. Respond with a brief explanation if needed, and then list the point coordinates in JSON format.
"""

post_prompt3 = """Your task is to identify specific points in the image based on the question. Respond with a brief explanation if needed, followed by a list of 2D point coordinates.
Each point should be represented as a normalized (x, y) tuple, where both x and y values are integers between 0 and 1000, corresponding to the position within the image (e.g., for a point at pixel (50, 75) in a 100*100 image, the normalized coordinate is (500, 750)).
Format your final answer strictly as follows on the last line of your response:
Answer: [(x1, y1), (x2, y2), ..., (xn, yn)]
Do not include additional text after this line.
"""

post_prompt4 = """Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should range from 0 to 1000, representing the relative pixel positions of the points in the image."""

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt=post_prompt4,
    ),
    config=config,
    name="Where2Place",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate_qwen3vl.py")
