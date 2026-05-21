config = dict(
    dataset_path="/code1/data/robobrain2-benchmark/VSI-Bench",
    split="test",
    processed_dataset_path="/code1/data/robobrain2-benchmark/VSI-Bench",
    processor="process.py",
)

MCA_QUESTION_TYPES = set(
    [
        "object_rel_direction_easy",
        "object_rel_direction_medium",
        "object_rel_direction_hard",
        "object_rel_distance",
        "route_planning",
        "obj_appearance_order",
    ]
)
NA_QUESTION_TYPES = set(
    [
        "object_abs_distance",
        "object_counting",
        "object_size_estimation",
        "room_size_estimation",
    ]
)

pre_prompt1 = "These are frames of a video.\n"
pre_prompt2 = "Carefully watch the video.\n"


def post_prompt1(question_type: str, **kwargs) -> str:
    prompt = "Carefully analyze the question above and reason through it step by step. You must conclude your response with a line in the following format:"
    # prompt = "Carefully analyze the question above and reason through it step by step. Conclude your response with a line in the following format:"
    if question_type in MCA_QUESTION_TYPES:
        return f"{prompt}\nAnswer: $LETTER (without quotes), where $LETTER corresponds to the correct option."
        # return f"{prompt}\nAnswer with the option's letters from the given choices directly. The last line of your response must be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of the options."
    elif question_type in NA_QUESTION_TYPES:
        return f"{prompt}\nAnswer: $NUMBER (without quotes), where $NUMBER is a number (integer or float) corresponds to the correct answer."
        # return f"{prompt}\nThe last line of your response must be of the following format: 'Answer: $NUMBER' (without quotes), where $NUMBER is a number (integer or float) corresponds to the correct answer."
    else:
        raise ValueError(f"Unknown question type: {question_type}")

def post_prompt2(question_type: str, **kwargs) -> str:
    if question_type in MCA_QUESTION_TYPES:
        return f"The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes), where $LETTER corresponds to the correct option."
    elif question_type in NA_QUESTION_TYPES:
        return f"The last line of your response should be of the following format: 'Answer: $NUMBER' (without quotes), where $NUMBER is a number (integer or float) corresponds to the correct answer."
    else:
        raise ValueError(f"Unknown question type: {question_type}")
    
post_prompt3 = """"""

dataset = dict(
    type="VideoDataset",
    config=config,
    anno_file="data.json",
    prompt_template=dict(
        type="PromptTemplate", pre_prompt=pre_prompt2, post_prompt=post_prompt3
    ),
    name="vsi_bench_test",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")

