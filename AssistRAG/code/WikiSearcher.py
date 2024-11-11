import os
import pickle
import faiss
from typing import List
import numpy as np
import pandas as pd
import torch
import json
from pyserini.search.lucene import LuceneSearcher
from datasets import load_from_disk, Dataset

import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from transformers import AutoModelForSequenceClassification, AutoTokenizer, BatchEncoding, PreTrainedTokenizerFast
from transformers.modeling_outputs import SequenceClassifierOutput


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def batch_encode(tokenizer: PreTrainedTokenizerFast, query: str, texts: List[str], titles: List[str]) -> dict:
    text_pairs = ['{}: {}'.format(title, text) for title, text in zip(titles, texts)]
    return tokenizer([query] * len(texts),  # Repeat the query for each text pair
                     text_pair=text_pairs,
                     max_length=192,
                     padding='longest',  # Pad to the longest sequence in the batch
                     truncation=True,
                     return_tensors='pt')

class WikiSearcher:
    def __init__(self,
                 index_path='/wikipedia-dpr-100w'):
        self.index_path = index_path
        self.searcher = LuceneSearcher(index_path)
        self.tokenizer = AutoTokenizer.from_pretrained('/data/webgpt/models/simlm-msmarco-reranker')
        self.model = AutoModelForSequenceClassification.from_pretrained('/data/webgpt/models/simlm-msmarco-reranker')

        # Move model to GPU if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if self.device.type == "cuda":
            self.model.half()  # convert to half precision

    def search(self, query, num1, num2):
        hits = self.searcher.search(query,num1)
        
        all_contents = [json.loads(self.searcher.doc(hits[i].docid).raw())['contents'] for i in range(num1)]
        all_title = []
        all_text = []
        for contents in all_contents:
            title = contents.split("\n")[0].strip("\"")
            text = "\n".join(contents.split("\n")[1:])
            all_title.append(title)
            all_text.append(text)
        all_data = Dataset.from_dict({"title": all_title,"text":all_text})

        logits = []

        # Process data in batches
        batch_size = 100
        with torch.no_grad():
            for i in range(0, len(all_data['title']), batch_size):
                batch_text = all_data['text'][i:i+batch_size]
                batch_title = all_data['title'][i:i+batch_size]
                batch_dict = batch_encode(self.tokenizer, query, batch_text, batch_title)
                
                # Move batch to GPU
                batch_dict = {key: val.to(self.device) for key, val in batch_dict.items()}

                outputs: SequenceClassifierOutput = self.model(**batch_dict, return_dict=True)
                logits.extend(outputs.logits[:, 0].cpu())

        use_ids = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)
        doc_list = []
        for i in range(num2):
            s = all_data['title'][use_ids[i]] + ' -- ' + all_data['text'][use_ids[i]]
            doc_list.append(s)
        return doc_list

class MemorySearcher:
    def __init__(self,
                 index_path='/AssistRAG/dataset/hotpot/index',
                 dataset_path='/AssistRAG/dataset/hotpot/memory/final_output.json'):
        self.index_path = index_path
        self.searcher = LuceneSearcher(index_path)
        self.data = {}
        with open(dataset_path) as f:
            data = json.load(f)
            for line in data:
                self.data[line['question']] = line['target']

    def search(self, query, num):
        hits = self.searcher.search(query, num)
        ids = [e.docid for e in hits]
        doc_list = []
        for idx in ids:
            s = 'question: ' + idx + '\nanswer: ' + self.data[idx]
            doc_list.append(s)
        return doc_list
