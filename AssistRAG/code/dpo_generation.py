import os
import re
import datetime
import shutil
import torch
import openai
import json
import argparse
from tqdm import tqdm
from utils import load_dataset, get_final_answer_from_pred, normalize_answer, loading_model, f1_score
from evaluate import evaluate
from generator import generator
from assistant import assistant
from WikiSearcher import WikiSearcher, MemorySearcher
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoTokenizer, AutoModel

def generate_pairs(args):
    model, tokenizer = loading_model(args)
    external_retriever = WikiSearcher()
    #generator_gpt = generator('gpt-35-turbo')
    #generator_llama = generator('llama', model, tokenizer) 
    assistant_glm = assistant('chatglm', model, tokenizer)

    f = json.load(open(args.dataset_path))
    outputs = []
    for line in tqdm(f[800:3000]):
        question = line['question']
        answer = re.search(r'Answer:(.*?)Supporting Facts:', line['input'], re.DOTALL).group(1).strip()
        external_retrieval_results = external_retriever.search(question, 100, 5)
        if len(question.split())>100:
            continue
        try:
            reference = assistant_glm.knowledge_extraction(question, "\n".join(external_retrieval_results))
            cot = assistant_glm.cot_answer(question)
        except:
            continue
        format_dpo = {}
        format_dpo['instruction'] = 'Please extract relevant snippets from search results that would be helpful in answering this question.\nQuestion: {question}.\nSearch results: {search_results}\nsnippets: '.format(question=question, search_results="\n".join(external_retrieval_results))
        format_dpo['input'] = ""
        format_dpo['output'] = [reference, cot]
        format_dpo['question'] = question
        format_dpo['answer'] = answer
        outputs.append(format_dpo)
    with open(args.output_path,"w",encoding="utf-8") as fo:
        json.dump(outputs, fo, indent=4)

def preference_alignment(args):
    args.model_name = args.llm_name
    model, tokenizer = loading_model(args)
    generator_llama = generator('llama', model, tokenizer)
    f = json.load(open(args.dataset_path))
    outputs = []
    for line in tqdm(f):
        format_dpo = {}
        format_dpo['instruction'] = line['instruction']
        format_dpo['input'] = line['input']
        question = line['question']
        answer = line['answer']
        reference_snippets = line['output'][0]
        reference_cot = line['output'][1]

        try:
            output_snippets = generator_llama.rag_answer(question, reference_snippets)
            output_cot = generator_llama.rag_answer(question, reference_cot)
        except:
            continue
        reward_snippets = f1_score(output_snippets, answer)
        reward_cot = f1_score(output_cot, answer)
        if reward_snippets >= reward_cot:
            format_dpo['output'] = [reference_snippets, reference_cot]
        else:
            format_dpo['output'] = [reference_cot, reference_snippets]
        outputs.append(format_dpo)
    with open(args.output_path,"w",encoding="utf-8") as fo:
        json.dump(outputs, fo, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str, default='/data/webgpt/models/', help="Path to the config file")
    parser.add_argument("--model_name", type=str, default='assistllm', help="Path to the config file")
    parser.add_argument("--llm_name", type=str, default='llama-2-13b-chat', help="Path to the config file")
    parser.add_argument("--dataset_path", type=str, default='/data/webgpt/IIA/dataset/hotpot/memory/train_split_0', help="Path to the config file")
    parser.add_argument("--output_path", type=str, default='/data/webgpt/IIA/data/dpo/dpo.json', help="Path to the config file")
    parser.add_argument("--mode", type=str, default='generate_pairs', help="Path to the config file")
    args = parser.parse_args()
    if args.mode == 'generate_pairs':
        generate_pairs(args)
    else:
        preference_alignment(args)