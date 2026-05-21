config = dict(
    dataset_path="/code1/data/robobrain2-benchmark/ERQA",
    split="test",
    processed_dataset_path="/code1/data/robobrain2-benchmark/ERQA",
    processor="process.py",
)

post_prompt1 = """Carefully analyze the multiple-choice question above and reason through it step by step. Conclude your response with a line in the following format: Answer: $LETTER (without quotes), where $LETTER is the letter of the correct choice."""

post_prompt2 = """Carefully analyze the multiple-choice question above and reason through it step by step.
Use at most 10 sentences for your reasoning. Do NOT restate or repeat the question.
After the reasoning, output exactly one line in the following format:
Answer: $LETTER, where $LETTER is the letter of the correct choice.
"""

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

