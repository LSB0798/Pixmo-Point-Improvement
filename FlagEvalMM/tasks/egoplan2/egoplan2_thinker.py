config = dict(
    dataset_path="/data/lishuaibing/test_process/robobrain2-benchmark/EgoPlan-Bench2",
    split="test",
    # split="test_wjy",
    processed_dataset_path="/data/lishuaibing/test_process/robobrain2-benchmark/EgoPlan-Bench2",
    processor="process.py",
)

pre_prompt = """
You are an egocentric video planner. \
Decide the SINGLE next atomic action AFTER the last observed frame. \
Rules: \
- Obey chronological order. Prefer the action that should happen immediately NEXT, not the final goal. \
- If a tool/valve/door was turned ON or OPENED, consider turning it OFF or CLOSING before moving away. \
- Avoid repeating actions that already happened in the observed timeline unless repetition is the natural cycle. \
- Prefer safety/cleanup (turn off tap/close oven/wipe hands) before continuing to cook. \
- Use object preconditions (attach before set, pick up before place, remove hand before flatten/cut).
"""

pre_prompt2 = """According to video and ending frame. """

pre_prompt3 = """"""

post_prompt = """"""

dataset = dict(
    type="VideoDataset",
    prompt_template=dict(
        type="PromptTemplate",
        pre_prompt=pre_prompt3,
        post_prompt=post_prompt,
    ),
    config=config,
    name="egoplan2",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
