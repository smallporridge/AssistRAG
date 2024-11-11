import os
import datetime
import shutil
import torch
import openai
import json
import argparse
from tqdm import tqdm
from utils import load_dataset, get_final_answer_from_pred, normalize_answer, loading_model
from evaluate import evaluate
from generator import generator
from assistant import assistant
from WikiSearcher import WikiSearcher, MemorySearcher
from WikiSearcher_llm import WikiSearcher_llm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoTokenizer, AutoModel

def deduplication(search_results):
    results = []
    for i in search_results:
        if not i in results:
            results.append(i)
    return "\n".join(results)

def generate_assistance(args):
    dataset_path = "" #your dataset_path. Format {"question": xxx, "answer": xxx}
    model, tokenizer = loading_model(args)
    assistant_glm = assistant('chatglm', model, tokenizer)

    # external_retriever = WikiSearcher()
    self_retriever = MemorySearcher()
    external_retriever = WikiSearcher_llm()
    
    dataset = load_dataset(dataset_path)
    is_2wiki = (args.dataset_name == "2WikiMultihopQA")
    outputs = []
    for item in tqdm(dataset[:500]):
        question = item['question']
        answer = item['answer']
        external_retrieval_results = []
        

        queries = assistant_glm.question_analysis(question).split('\n')

        for query in queries[:2]:
            external_retrieval_results.extend(external_retriever.search(query, 5))
        reference = assistant_glm.knowledge_extraction(question, deduplication(external_retrieval_results))

        self_retrieval = self_retriever.search(question, 3)
        memory = "\n\n".join(self_retrieval)
        
        format_result = {}
        format_result['question'] = question
        format_result['queries'] = queries
        format_result['reference'] = reference
        format_result['memory'] = memory
        format_result['answer'] = answer
        outputs.append(format_result)
    with open(args.temporal_path,"w",encoding="utf-8") as fo:
        json.dump(outputs, fo, indent=4)

def generate_answer(args):
    assistant = args.model_name
    main_llm = args.llm_name
    if args.llm_name == 'gpt-35-turbo':
        generator_llm = generator('gpt-35-turbo')
    else:
        args.model_name = args.llm_name
        model, tokenizer = loading_model(args)
        generator_llm = generator('llama', model, tokenizer)
    
    dataset = json.load(open(args.temporal_path))
    is_2wiki = (args.dataset_name == "2WikiMultihopQA")
    outputs = []
    em, f1, precision, recall = 0.0, 0.0, 0.0, 0.0

    for item in tqdm(dataset):
        question = item['question']
        answer = item['answer']
        reference = item['reference']
        memory = item['memory']

        if args.mode == "full":
            final_output = generator_llm.full_answer(question, reference, memory)
            # anchor_text = ['answer is (.*?)\.']
            # extract_answer = get_final_answer_from_pred(final_output, anchor_text)
        elif args.mode == "rag":
            final_output = generator_llm.rag_answer(question, reference)
        elif args.mode == "direct":
            final_output = generator_llm.direct_answer(question)
        else:
            break
        extract_answer = final_output
        eval_result = evaluate(question=question, 
                                pred=extract_answer, 
                                answer=answer, 
                                is_2wiki=is_2wiki)
        em += eval_result['em']
        f1 += eval_result['f1']
        precision += eval_result['precision']
        recall += eval_result['recall']
        
        item['predict'] = extract_answer
        item['final_output'] = final_output
        item['eval_result'] = eval_result
        outputs.append(item)
    with open(args.output_path,"w",encoding="utf-8") as fo:
        json.dump(outputs, fo, indent=4)

    print("em:" + str(em/len(dataset))+'\n'+"f1:" + str(f1/len(dataset))+'\n'+"precision:" + str(precision/len(dataset))+'\n'+"recall:" + str(recall/len(dataset))+'\n')
    with open(args.evaluate_path,"a",encoding="utf-8") as fe:
        data = [assistant, main_llm, args.mode, str(em/len(dataset)), str(precision/len(dataset)), str(recall/len(dataset)), str(f1/len(dataset))]
        fe.write(" | ".join(data)+'\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str, default='/data/webgpt/models/', help="Path to the config file")
    parser.add_argument("--model_name", type=str, default='assistllm', help="Path to the config file")
    parser.add_argument("--llm_name", type=str, default='llama-2-13b-chat', help="Path to the config file")
    parser.add_argument("--dataset_name", type=str, default='hotpot', help="Path to the config file")
    parser.add_argument("--temporal_path", type=str, default='/AssistRAG/dataset/hotpot/results/temporal.json', help="Path to the config file")
    parser.add_argument("--output_path", type=str, default='/AssistRAG/dataset/hotpot/results/results.json', help="Path to the config file")
    parser.add_argument("--evaluate_path", type=str, default='/AssistRAG/dataset/hotpot/evaluation/evaluation.txt', help="Path to the config file")
    parser.add_argument("--stage", type=str, default='assistance', help="Path to the config file")
    parser.add_argument("--mode", type=str, default='full', help="Path to the config file")
    args = parser.parse_args()
    #dataset_name = "hotpot" # # 2WikiMultihopQA/hotpot/bamboogle/
    if args.model_name == "chatglm3-6b-dpo":
        args.model_name = "chatglm3-6b-dpo/checkpoint-344"
    if args.stage == 'assistance':
        generate_assistance(args)
    else:
        generate_answer(args)

