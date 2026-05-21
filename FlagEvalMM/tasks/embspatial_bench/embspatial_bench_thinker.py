config = dict(
    dataset_path="/code1/data/robobrain2-benchmark/EmbSpatial-Bench",
    split="test",
    processed_dataset_path="/code1/data/robobrain2-benchmark/EmbSpatialBench",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate"),
    config=config,
    name="embspatial_bench",
)

evaluator = dict(type="BaseEvaluator", detailed_keys=["sub_task"])
