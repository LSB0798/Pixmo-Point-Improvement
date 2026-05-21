config = dict(
    dataset_path="/code1/data/OneThinker-eval",
    split="",
    processed_dataset_path="/code1/data/OneThinker-eval",
    processor="process.py",
)

post_prompt = """Your answer should be formatted as a list of tuples, i.e. [[(x1, y1), (x2, y2), ...], ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should range from 0 to 1000, representing the relative pixel positions of the points in the image."""

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt=post_prompt,
    ),
    config=config,
    anno_file="eval_seg_refcoco_flagevalmm.json",
    name="seg_refcoco",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate_seg_thinker.py")
