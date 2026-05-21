config = dict(
    dataset_path="/code1/data/robobrain2-benchmark/RefCOCOplus",
    split="val",
    processed_dataset_path="/code1/data/robobrain2-benchmark/RefCOCOplus",
    processor="process.py",
)

post_prompt = """The bbox coordinate format is [top-left x, top-left y, bottom-right x, bottom-right y].All values must be integer points between 0 and 1, inclusive."""

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt=post_prompt,
    ),
    config=config,
    name="refcoco_plus_val",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
