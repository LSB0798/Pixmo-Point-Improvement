config = dict(
    dataset_path="/data/lishuaibing/test_process/OneThinker-eval",
    split="",
    processed_dataset_path="/data/lishuaibing/test_process/OneThinker-eval",
    processor="process.py",
)

post_prompt = """The bbox coordinate format is [top-left x, top-left y, bottom-right x, bottom-right y].All values must be integer points between 0 and 1000, inclusive."""

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt=post_prompt,
    ),
    config=config,
    anno_file="eval_refcoco_val_flagevalmm.json",
    name="refcoco_val",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate_qwen3vl.py")
