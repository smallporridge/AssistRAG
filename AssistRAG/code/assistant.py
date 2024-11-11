import os
import torch
import openai
import re

@torch.inference_mode()
def get_llama_answer(model, tokenizer, instruction_qa, max_length):
    inputs = tokenizer([instruction_qa],return_token_type_ids=False,return_tensors="pt", max_length=max_length)
    inputs = {k: torch.tensor(v).to('cuda') for k, v in inputs.items()}
    output_ids = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.5,
        max_new_tokens=max_length, 
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True
    )
    output_ids = output_ids[0][inputs['input_ids'].size(1):]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True,
    )
    return outputs.strip()

@torch.inference_mode()
def get_llm_answer(model, tokenizer, instruction_qa):
    inputs = tokenizer([instruction_qa],return_token_type_ids=False,return_tensors="pt")
    inputs = {k: torch.tensor(v).to('cuda') for k, v in inputs.items()}
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.pad_token_id,
        num_beams=1,
        do_sample=False,
        temperature=1.0,
        use_cache=True
    )
    output_ids = output_ids[0][inputs['input_ids'].size(1):]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True,
    )
    return outputs.strip()



class assistant:
    def __init__(self, llm, model=None, tokenizer=None):
        self.llm = llm
        if llm == "gpt-35-turbo":
            openai.api_type = "azure"
            openai.api_base = "https://baaiks.openai.azure.com/"
            openai.api_version = "2023-05-15"
            openai.api_key = "" #API-KEY
        else:
            self.model = model
            self.tokenizer = tokenizer


    def get_llm_output(self,prompt,system_prompt="I want you to act as an LLM helper."):
        if self.llm == 'gpt-35-turbo':
            response = openai.ChatCompletion.create(
                engine=self.llm,
                messages=[{"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}],
                temperature=0
            )
            output = response['choices'][0]['message']['content']
            return output
        elif "llama" in self.llm or "vicuna" in self.llm:
            output = get_llama_answer(self.model, self.tokenizer, prompt, max_length=500)
            return output
        elif "mistral" in self.llm:
            prompt = "<s>[INST]" + prompt + "[/INST]"
            output = get_llama_answer(self.model, self.tokenizer, prompt, max_length=500)
            return output
        else:
            output = get_llm_answer(self.model, self.tokenizer, prompt)
            return output

    def question_analysis(self, question): 
        prompt = 'Please generate a series of search queries that can be used to find information relevant to the given question.\nQuestion: {question}.\nSearch queries: '
        prompt = prompt.format(question=question)
        output = self.get_llm_output(prompt)
        return str(output)
    
    def memory_construction(self, question, supporting_facts, answer):
        prompt = "You are given 1) the question, 2) the answer, 3) the supporting facts where the answer can be derived. You are supposed to figure out the reasoning process towards the answer step-by-step without other content. Be concise and direct. \nQuestion: {question}\nAnswer: {answer}\nSupporting Facts: {supporting_facts}\nreasoning: "
        prompt = prompt.format(question=question,answer=answer,supporting_facts=supporting_facts)
        output = self.get_llm_output(prompt)
        return str(output)

    def knowledge_extraction(self, question, search_results):
        prompt = 'Please extract relevant snippets from search results that would be helpful in answering this question.\nQuestion: {question}.\nSearch results: {search_results}\nsnippets: '
        prompt = prompt.format(question=question, search_results=search_results)
        output = self.get_llm_output(prompt)
        return str(output)
    
    def cot_answer(self, question): 
        qa_prompt = """Please answer this multi-hop question step-by-step.\nquestion: {question}\nAnswer:"""
        qa_prompt = qa_prompt.format(question=question)
        output = self.get_llm_output(qa_prompt,system_prompt="You are a helpful query answering assistant.")
        return str(output)
