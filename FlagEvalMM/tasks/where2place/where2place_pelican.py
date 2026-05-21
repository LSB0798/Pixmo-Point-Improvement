config = dict(
    dataset_path="/code1/data/robobrain2-benchmark/Where2Place",
    split="test",
    processed_dataset_path="/code1/data/robobrain2-benchmark/Where2Place/",
    processor="process.py",
)

post_prompt1 = """"""

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt=post_prompt1,
    ),
    config=config,
    name="Where2Place",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")

