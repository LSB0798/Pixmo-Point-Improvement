task_name = "vqa"

config = dict(
    dataset_path="/code1/data/robobrain2-benchmark/BLINK",
    split="val",
    processed_dataset_path="/code1/data/robobrain2-benchmark/BLINK",
    processor="process.py",
    dataset_names=[
        # "Art_Style",
        # "Counting",
        # "Forensic_Detection",
        # "Functional_Correspondence",
        # "IQ_Test",
        # "Jigsaw",
        # "Multi-view_Reasoning",
        # "Object_Localization",
        "Relative_Depth",
        # "Relative_Reflectance",
        # "Semantic_Correspondence",
        "Spatial_Relation",
        # "Visual_Correspondence",
        # "Visual_Similarity",
    ],
)

post_prompt1 = """"""

post_prompt2 = """The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options."""

dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt=post_prompt1,
    ),
    name="blink_val",
)

evaluator = dict(type="BaseEvaluator", detailed_keys=["sub_task"])

