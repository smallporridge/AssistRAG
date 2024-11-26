import dataset
import argparse
import re
import torch
import os
import json
from datasets import load_from_disk
import openai
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoTokenizer, AutoModel
from fastchat.model import load_model
from tqdm import tqdm


def get_openai_answer(model, instruction_qa):
    response = openai.ChatCompletion.create(
        engine=model,
        messages=[{"role": "user", "content": instruction_qa}],
        temperature=0.5,
    )
    return response["choices"][0]["message"]["content"]


@torch.inference_mode()
def get_llama_answer(model, tokenizer, instruction_qa, max_length):
    inputs = tokenizer(
        [instruction_qa],
        return_token_type_ids=False,
        return_tensors="pt",
        max_length=max_length,
    )
    inputs = {k: torch.tensor(v).to("cuda") for k, v in inputs.items()}
    output_ids = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.5,
        max_new_tokens=256,
        num_return_sequences=1,
    )
    output_ids = output_ids[0][inputs["input_ids"].size(1) :]
    outputs = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
    )
    return outputs.strip()


def get_falcon_answer(model, tokenizer, instruction_qa):
    sequences = model(
        instruction_qa,
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    return sequences[0]["generated_text"]


def get_chatglm_answer(model, tokenizer, instruction_qa):
    response, history = model.chat(tokenizer, instruction_qa, history=[])
    return response.strip()


def loading_model(args):
    model_path = args.model_folder + args.model_name
    if "chatglm" in model_path or "checkpoint" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = (
            AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
        )
        model = model.eval()
    elif "llama" in model_path or "vicuna" in model_path:
        model, tokenizer = load_model(model_path, "cuda", 1)
    elif "falcon" in model_path:
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
        model = "gpt-35-turbo"
        tokenizer = ""
    return model, tokenizer


def process_memory():
    outputs_train = []
    outputs_dev = []
    path = "/data/webgpt/reason/prompt_compress/data/hotpot_qa/train"
    dataset = load_from_disk(path)
    prompt = "You are given 1) the question, 2) the answer, 3) the supporting facts where the answer can be derived. You are supposed to figure out the reasoning process towards the answer step-by-step without other content. Be concise and direct. \nQuestion: {question}\nAnswer: {answer}\nSupporting Facts: {supporting_facts}\nreasoning: "
    i = 0
    num = 0
    for line in tqdm(dataset):
        question = line["question"]
        answer = line["answer"]
        supporting_facts = ""
        try:
            for title, sent_id in zip(
                line["supporting_facts"]["title"], line["supporting_facts"]["sent_id"]
            ):
                idx = line["context"]["title"].index(title)
                supporting_facts += line["context"]["sentences"][idx][sent_id] + "\n"
        except:
            continue

        user_prompt = prompt.format(
            question=question, answer=answer, supporting_facts=supporting_facts
        )
        format_train = {}
        format_train["question"] = question
        format_train["input"] = user_prompt
        format_train["target"] = ""
        outputs_train.append(format_train)
        i += 1

        if i % 11313 == 0:
            with open(
                "/data/webgpt/IIA/dataset/hotpot/memory/train_split_" + str(num),
                "w",
                encoding="utf-8",
            ) as fo:
                json.dump(outputs_train, fo, indent=4)
                num += 1
                outputs_train = []
    with open(
        "/data/webgpt/IIA/dataset/hotpot/memory/train_split_" + str(num),
        "w",
        encoding="utf-8",
    ) as fo:
        json.dump(outputs_train, fo, indent=4)


def memory_construct(args):
    model, tokenizer = loading_model(args)
    candidates = json.load(open(args.data_path))
    outputs = []
    for item in tqdm(candidates):
        prompt = item["input"]
        item["target"] = get_chatglm_answer(model, tokenizer, prompt)
        outputs.append(item)
    with open(args.output_path, "w", encoding="utf-8") as fo:
        json.dump(outputs, fo, indent=4)


def memory_indexing(args):
    candidates = json.load(open(args.data_path))
    outputs = []
    for item in candidates:
        data = {}
        data["id"] = item["question"]
        data["contents"] = item["question"]
        outputs.append(data)
    with open(args.output_path, "w", encoding="utf-8") as fo:
        json.dump(outputs, fo, indent=4)


def merge_json(folder_path):
    # List all files that start with 'output_' and end with '.json'
    files_to_merge = [
        f
        for f in os.listdir(folder_path)
        if f.startswith("output_llama") and f.endswith(".json")
    ]

    # Initialize an empty list to hold the merged data
    merged_data = []

    # Iterate through each file and merge the data
    for file in files_to_merge:
        with open(os.path.join(folder_path, file), "r") as f:
            data = json.load(f)
            merged_data.extend(data)

    # Optionally, save the merged data to a new file
    with open("/data/webgpt/IIA/data/dpo/dpo_llama.json", "w") as f:
        json.dump(merged_data, f, indent=4)

    # Print the number of items in the merged data
    print(f"Merged {len(merged_data)} items from {len(files_to_merge)} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_folder", type=str, default="/models/", help="Path to the config file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="checkpoint-768",
        help="Path to the config file",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/AssistRAG/dataset/hotpot/train.json",
        help="Path to the config file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/AssistRAG/dataset/hotpot/memory.json",
        help="Path to the config file",
    )
    args = parser.parse_args()
    # memory_construct(args)
    data_path = "/data/webgpt/IIA/data/dpo"
    merge_json(data_path)
