import json
import re
import string
from typing import List, Dict, Any, Tuple, Union, Set
import tiktoken
import openai
from fastchat.model import load_model
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoTokenizer, AutoModel

def load_dataset(dataset_path):
    with open(dataset_path,"r",encoding="utf-8") as f:
        data = json.load(f) 
    data = [{"question": item['question'],
            "answer": item['answer']}  for item in data]
    return data 


def num_tokens(text):
    return len(tiktoken.encoding_for_model("gpt-3.5-turbo").encode(text))

def format_ref(reference):
    def format_step(step: str) -> str:
        return step.strip('\n').strip().replace('\n', '  ')
    return format_step("\n".join(reference))

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_final_answer_from_pred(pred: str, anchor_text: List[str]):
    final_ans = []
    for at in anchor_text:
        find = re.compile(at).search(pred)
        if find:
            final_ans.append(find.group(1))
            break
    return ' '.join(final_ans).strip()

def loading_model(args):
    model_path = args.model_folder + args.model_name
    print("loading model: {model_name}".format(model_name=args.model_name))
    if 'chatglm' in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
        model = model.eval()
    elif 'llama' in model_path or 'vicuna' in model_path:
        model, tokenizer = load_model(model_path,'cuda', 1)
    elif 'mistral' in model_path:
        model = AutoModelForCausalLM.from_pretrained(model_path).half().cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = model.eval()

    elif 'falcon' in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = transformers.pipeline(
            "text-generation",
            model=model_path,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        openai.api_type = "azure"
        openai.api_base = "https://baaiks.openai.azure.com/"
        openai.api_version = "2023-05-15"
        openai.api_key = "488afb0795c94e5fad8b4df8dab246e2"
        model = 'gpt-35-turbo'
        tokenizer = ''
    return model, tokenizer

from collections import Counter

def f1_score(
    prediction: str,
    ground_truths: str
):
    final_metric = {'f1': 0, 'precision': 0, 'recall': 0}
    if isinstance(ground_truths,str):
        ground_truths = [ground_truths]
    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)
    
        if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
            continue
        if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
            continue
        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        for k in ['f1', 'precision', 'recall']:
            final_metric[k] = max(eval(k), final_metric[k])
    return final_metric['f1']

def recall_score(
    prediction: str,
    ground_truths: str
):
    final_metric = {'f1': 0, 'precision': 0, 'recall': 0}
    if isinstance(ground_truths,str):
        ground_truths = [ground_truths]
    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)
    
        if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
            continue
        if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
            continue
        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        for k in ['f1', 'precision', 'recall']:
            final_metric[k] = max(eval(k), final_metric[k])
    return final_metric['recall']

def best_subspan_em(prediction: str, ground_truths) -> float:
    if isinstance(ground_truths,str):
        ground_truths = [ground_truths]
    normalized_prediction = normalize_answer(prediction)
    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)

        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0

def calculate_em(prediction, ground_truths):
    if isinstance(ground_truths,str):
        ground_truths = [ground_truths]
    normalized_prediction = normalize_answer(prediction)
    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() == normalized_prediction.lower():
            return 1.0
    return 0.0