config = dict(
    dataset_path="/code1/data/robobrain2-benchmark/CV-Bench",
    split="test",
    processed_dataset_path="/code1/data/robobrain2-benchmark/CV-Bench",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate"),
    config=config,
    name="cv_bench_test",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
