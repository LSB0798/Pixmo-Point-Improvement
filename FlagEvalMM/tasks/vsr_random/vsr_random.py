config = dict(
    dataset_path="/code1/data/robobrain2-benchmark/vsr_random",
    split="test",
    processed_dataset_path="/code1/data/robobrain2-benchmark/vsr_random",
    processor="process.py",
)

post_prompt1 = """Your task is to answer the question above. \n\nFormat your final answer strictly as follows: yes or no\n\nDo not include additional text after this line."""

post_prompt2 = """Answer with a single word yes or no."""

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt=post_prompt2,
    ),
    config=config,
    name="vsr_random",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")

