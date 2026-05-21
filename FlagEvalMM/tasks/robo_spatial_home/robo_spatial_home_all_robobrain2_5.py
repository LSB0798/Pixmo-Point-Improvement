config = dict(
    dataset_path="/code1/data/robobrain2-benchmark/RoboSpatial-Home",
    split="all",
    processed_dataset_path="/code1/data/robobrain2-benchmark/RoboSpatial-Home",
    processor="process.py",
)


post_prompt_point = """Please provide its 2D coordinates. Your answer should be formatted as a tuple, i.e. [(x, y)], where the tuple contains the x and y coordinates of a point satisfying the conditions above."""

post_prompt_yes_no = """(A) Yes (B) No. Please output the answer in the following format: (A) or (B)"""


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

evaluator = dict(type="BaseEvaluator", eval_func="robo_spatial_evaluate_robobrain2_5.py")