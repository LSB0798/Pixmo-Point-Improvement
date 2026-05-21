config = dict(
    dataset_path="/code1/data/robobrain2-benchmark/ERQA",
    split="test",
    processed_dataset_path="/code1/data/robobrain2-benchmark/ERQA",
    processor="process.py",
)

post_prompt1 = """Please answer directly with only the letter of the correct option and nothing else."""

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt=post_prompt1,
    ),
    config=config,
    name="erqa",
)

evaluator = dict(type="BaseEvaluator", detailed_keys=["sub_task"])

