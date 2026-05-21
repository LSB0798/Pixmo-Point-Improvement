config = dict(
    dataset_path="/code1/data/robobrain2-benchmark/SAT",
    split="test",
    processed_dataset_path="/code1/data/robobrain2-benchmark/SAT",
    processor="process.py",
)

post_prompt1 = """Please only answer with the option letter."""

post_prompt2 = """"""

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt=post_prompt1,
    ),
    config=config,
    name="SAT",
)

evaluator = dict(type="BaseEvaluator", detailed_keys=["sub_task"])