config = dict(
    dataset_path="/code1/data/robobrain2-benchmark/ShareRobot-Bench/",
    # dataset_path="/data/nlp/nlp_team1/data/ShareRobot-Bench",
    split="test",
    processed_dataset_path="/code1/data/robobrain2-benchmark/ShareRobot-Bench/affordance/",
    # processed_dataset_path="/data/nlp/nlp_team1/data/ShareRobot-Bench/affordance/",
    processor="process.py",
)

post_prompt1 = "Please generate a set of bounding box (bbox) coordinates based on the image and description.The bbox coordinate format is [top-left x, top-left y, bottom-right x, bottom-right y].All values must be integer points between 0 and 1000, inclusive."

post_prompt2 = "Only return the box in format: [x1, y1, x2, y2] with no other output, where both x and y values are floats between 0 and 1, corresponding to the position within the image."

post_prompt3 = "Only return the normalized box in format: [x1, y1, x2, y2] with no other output, where x1, y1, x2, y2 are normalized floating point numbers between 0 and 1, corresponding to the position within the image (e.g., for a box at pixel [50, 50, 75, 75] in a 100*100 image, the normalized coordinate is [0.5, 0.5, 0.75, 0.75])."

post_prompt4 = "Only return the box in format: [x1, y1, x2, y2] with no other output, where both x and y values are floats between 0 and 1, representing normalized coordinates corresponding to the position within the image."

dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt=post_prompt1,
    ),
    name="share_robot_affordance",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")

