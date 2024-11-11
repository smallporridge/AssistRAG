import os
import torch
import openai
import re
import time


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
        pad_token_id=tokenizer.eos_token_id,
    )
    output_ids = output_ids[0][inputs["input_ids"].size(1) :]
    outputs = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
    )
    return outputs.strip()


class generator:
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

    def get_llm_output(
        self, prompt, system_prompt="I want you to act as a question answering agent."
    ):
        if self.llm == "gpt-35-turbo":
            for i in range(3):
                try:
                    response = openai.ChatCompletion.create(
                        engine=self.llm,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0,
                    )
                    output = response["choices"][0]["message"]["content"]
                    return output
                except:
                    time.sleep(3)
                    continue
        elif "llama" in self.llm or "vicuna" in self.llm:
            output = get_llama_answer(
                self.model, self.tokenizer, prompt, max_length=1500
            )
            return output
        elif "mistral" in self.llm:
            prompt = "<s>[INST]" + prompt + "[/INST]"
            output = get_llama_answer(
                self.model, self.tokenizer, prompt, max_length=1500
            )
            return output
        else:
            response, history = self.model.chat(self.tokenizer, prompt, history=[])
            return response.strip()

    def full_answer(self, question, reference, memory):
        qa_prompt = """Examples: {memory}\n\nPlease answer the question based on the provided reference. Output a short answer with one word or phrase without other content. \nquestion: {question}\nreference: {reference}\nAnswer: """
        qa_prompt = qa_prompt.format(
            memory=memory, question=question, reference=reference
        )
        output = self.get_llm_output(
            qa_prompt, system_prompt="You are a helpful query answering assistant."
        )
        return str(output)

    def rag_answer(self, question, reference):
        qa_prompt = "Please answer the question based on the provided reference. Output a short answer with one word or phrase without other content.\nquestion: {question}\nreference: {reference}\nAnswer: "
        qa_prompt = qa_prompt.format(question=question, reference=reference)
        output = self.get_llm_output(
            qa_prompt, system_prompt="You are a helpful query answering assistant."
        )
        return str(output)

    def direct_answer(self, question):
        qa_prompt = """Please answer the question. Output a short answer with one word or phrase without other content.\nquestion: {question}\nAnswer: """
        qa_prompt = qa_prompt.format(question=question)
        output = self.get_llm_output(
            qa_prompt, system_prompt="You are a helpful query answering assistant."
        )
        return str(output)

    def cot_answer(self, question, reference, memory):
        qa_prompt = """question: {question}\nAnswer:"""
        qa_prompt = qa_prompt.format(
            memory=memory, question=question, reference=reference
        )
        output = self.get_llm_output(
            qa_prompt, system_prompt="You are a helpful query answering assistant."
        )
        return str(output)
