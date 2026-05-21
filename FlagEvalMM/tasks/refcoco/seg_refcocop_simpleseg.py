config = dict(
    dataset_path="/code1/data/OneThinker-eval",
    split="",
    processed_dataset_path="/code1/data/OneThinker-eval",
    processor="process.py",
)

post_prompt = """"""

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt=post_prompt,
    ),
    config=config,
    anno_file="eval_seg_refcocop_flagevalmm.json",
    name="seg_refcocop",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate_seg_simpleseg.py")