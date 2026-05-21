config = dict(
    dataset_path="/code1/data/robobrain2-benchmark/SAT",
    split="test",
    processed_dataset_path="/code1/data/robobrain2-benchmark/SAT",
    processor="process.py",
)

post_prompt1 = """Carefully analyze the multiple-choice question above and reason through it step by step. Conclude your response with a line in the following format: Answer: $LETTER (without quotes), where $LETTER is the letter of the correct choice."""

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
