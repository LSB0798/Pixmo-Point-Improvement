import re
from typing import List, Dict, Any

def strip_answer(answer):
    answer = re.sub("The", "", answer)
    answer = re.sub("If", "", answer)
    answer = re.sub("[INST]", "", answer)
    answer = re.sub("[/INST]", "", answer)
    answer = re.sub("<Img>", "", answer)
    answer = re.sub("</Img>", "", answer)
    answer = re.sub("<RichMediaReference>", "", answer)
    answer = re.sub("</RichMediaReference>", "", answer)
    answer = re.sub("<richmediareference>", "", answer)
    answer = re.sub("</richmediareference>", "", answer)
    answer = re.sub("<Answer>", "", answer)
    answer = re.sub("</Answer>", "", answer)
    answer = re.sub("<answer>", "", answer)
    answer = re.sub("</answer>", "", answer)
    answer = answer.strip()
    return answer

# 删除<RichMediaReference> <Answer>等标签
def strip_special_tags(answer):
    if isinstance(answer, str):
        answer = re.sub("<RichMediaReference>", "", answer)
        answer = re.sub("</RichMediaReference>", "", answer)
        answer = re.sub("<richmediareference>", "", answer)
        answer = re.sub("</richmediareference>", "", answer)
        answer = re.sub("<Answer>", "", answer)
        answer = re.sub("</Answer>", "", answer)
        answer = re.sub("<answer>", "", answer)
        answer = re.sub("</answer>", "", answer)
        answer = answer.strip()
    return answer

def strip_special_tags_for_predictions(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    对 List[Dict] 形式的 predictions 做标签清洗：
    - 如果 pred["answer"] 是 str，则调用 strip_special_tags
    - 如果 pred["answer"] 是 dict，则对其中每个 str 类型的 value 调用 strip_special_tags
    """
    for pred in predictions:
        ans = pred.get("answer", None)

        # 单答案：字符串
        if isinstance(ans, str):
            pred["answer"] = strip_special_tags(ans)

        # 多次推理：字典，key 是推理 id，value 是答案字符串
        elif isinstance(ans, dict):
            cleaned = {}
            for k, v in ans.items():
                if isinstance(v, str):
                    cleaned[k] = strip_special_tags(v)
                else:
                    cleaned[k] = v
            pred["answer"] = cleaned

        # 其他类型就先不动
    return predictions

def remove_special_characters(text):
    pattern = r"[-`\\【】\*\$、,，。.；;:：？\?！!\s\n\u4e00-\u9fff0-9①②③④⑤⑥⑦\[\]\<>a-z=\'\"\(\)\{\}]+"
    cleaned_text = re.sub(pattern, "", text)
    return cleaned_text


def process_multiple_choice(answer):
    answer = strip_answer(answer)
    
    # keep the last line
    answer = answer.split("\n")[-1]
    
    # 统一成大写，兼容 (a)/(A)/a./A) 等
    answer = answer.upper()
    key_words = [
        "boxed",
        "Answer:",
        "Answer is",
        "answer is",
        "option is",
        "Correct option",
        "correct option",
        "Answer",
        "answer",
        "故选",
        "选择",
        "正确选项为",
        "答案选",
        "答案为",
        "答案是",
        "因此",
        "答案",
    ]

    # 与上面的大小写统一：把关键词也转成大写再比对
    for key_word in key_words:
        key_word_upper = key_word.upper()
        if key_word_upper in answer:
            answer = answer.split(key_word_upper)[-1]
            break

    answer = remove_special_characters(answer)
    
    # pattern = r"[A-Z]"
    # matches = re.findall(pattern, answer)
    # return "".join(matches)

    # 只从 A-G 中取第一个匹配
    m = re.search(r"[A-Z]", answer)
    return m.group(0) if m else ""


def remove_unit(value):
    units = [
        "cm",
        "m",
        "km",
        "mm",
        "s",
        "h",
        "kg",
        "g",
        "l",
        "ml",
        "mol",
        "厘米",
        "米",
        "千米",
        "°",
        "毫米",
        "月",
        "秒",
        "小时",
        "克",
        "千克",
        "升",
        "毫升",
        "摩尔",
    ]
    unit_pattern = r"^(\d+)(?:" + "|".join(units) + ")$"
    match = re.match(unit_pattern, value)
    if match:
        return match.group(1)
    else:
        return value


def convert_circled_numbers(text):
    circled_numbers = {
        "①": "1",
        "②": "2",
        "③": "3",
        "④": "4",
        "⑤": "5",
        "⑥": "6",
        "⑦": "7",
        "⑧": "8",
        "⑨": "9",
        "⑩": "10",
    }
    for circled, number in circled_numbers.items():
        text = text.replace(circled, number)
    return text


def normalize_string(raw_answer):
    if "$" not in raw_answer:
        wrong_answer_words = ["\\times", "不对", "不正确", "×"]
        for word in wrong_answer_words:
            raw_answer = raw_answer.replace(word, "错误")
    raw_answer = re.sub(r"\\text\s*\{(.*?)\}", r"\1", raw_answer)
    replace_dict = {
        "√": "正确",
        "：": ":",
        "$": "",
        "（": "(",
        "）": ")",
        "，": ",",
        "。": ".",
        "变小": "减小",
        "变大": "增大",
        "路程": "距离",
        "\\pi": "π",
        "＞": ">",
        "＜": "<",
        "；": ";",
    }
    # write to convert characters like ①②③④ to 1234

    for k, v in replace_dict.items():
        raw_answer = raw_answer.replace(k, v)

    # Convert circled numbers to regular numbers
    raw_answer = convert_circled_numbers(raw_answer)

    strict_replace_dict = {
        "错": "错误",
        "对": "正确",
        "(F)": "F",
        "(T)": "T",
        "(正确)": "正确",
        "(错误)": "错误",
        "“T”": "T",
        "“F”": "F",
    }
    if raw_answer in strict_replace_dict:
        raw_answer = strict_replace_dict[raw_answer]

    key_words = [
        "Answer:",
        "Answer is",
        "answer is",
        "Answer",
        "answer",
        "答案为",
        "答案是",
        "解是",
        "解为",
        "答案",
        "结果",
        "为",
        "因此",
        " = ",
    ]
    # get text after key_word
    for key_word in key_words:
        if key_word in raw_answer:
            raw_answer = raw_answer.split(key_word)[-1]
            break
    raw_answer = raw_answer.strip()
    # remove leading :
    if raw_answer.startswith(":"):
        raw_answer = raw_answer[1:]
    if len(raw_answer) > 0 and raw_answer[-1] in [".", ",", ":", ";"]:
        raw_answer = raw_answer[:-1]
    raw_answer = remove_unit(raw_answer)
    return raw_answer.strip()
