config = dict(
    dataset_path="/code1/data/robobrain2-benchmark/RoboSpatial-Home",
    split="all",
    processed_dataset_path="/code1/data/robobrain2-benchmark/RoboSpatial-Home",
    processor="process.py",
)

post_prompt_point = """Your task is to identify specific points in the image based on the question. Respond with a brief explanation if needed, followed by a list of 2D point coordinates.

Do not include additional text after this line.
"""

post_prompt_yes_no = """Your task is to answer the question above. Respond with a brief explanation if needed, followed by a yes or no answer in the last line of your response.

Format your final answer strictly as follows on the last line of your response:
Answer: yes or no

Do not include additional text after this line.
"""


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

evaluator = dict(type="BaseEvaluator", eval_func="robo_spatial_evaluate.py")

