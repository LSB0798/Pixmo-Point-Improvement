import os

config = dict(
    dataset_path="/data/nlp/nlp_team1/data/robovqa/convert_data/val",
    split="val_my",
    processed_dataset_path="/data/nlp/nlp_team1/data/robovqa/convert_data/",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate", post_prompt=""),
    name="robovqa",
    config=config,
)

evaluator = dict(
    type="RoboVQAGPTEvaluator",
    eval_model_name=os.getenv("ROBOVQA_GRADER_MODEL", "gpt-4o-mini-2024-07-18"),
    use_llm_evaluator=True,
    use_cache=True,
    num_threads=int(os.getenv("ROBOVQA_GRADER_THREADS", "8")),
    base_url=os.getenv("FLAGEVAL_BASE_URL"),
    api_key=os.getenv("FLAGEVAL_API_KEY"),
)

