import datasets
from accelerate import Accelerator
from llm_embedder.src.retrieval import DenseRetriever

accelerator = Accelerator()

class WikiSearcher_llm:
    def __init__(self):
        self.retriever = DenseRetriever(
            # for dense retriever
            query_encoder="BAAI/llm-embedder",
            pooling_method=["cls"],
            dense_metric="cos",
            query_max_length=128,
            tie_encoders=True,
            dtype="fp16",
            accelerator=accelerator
        )
        self.retriever.index("/data/webgpt/IIA/code/llm_embedder/data/nq/corpus.json", output_dir="/data/webgpt/IIA/code/llm_embedder/data/outputs", load_index=True)
        self.corpus = datasets.load_dataset("json", data_files="/data/webgpt/IIA/code/llm_embedder/data/nq/corpus.json", split="train")

    def search(self, query, hits):
        query_instruction = "Represent this query for retrieving relevant documents: "

        scores, indices = self.retriever.search(query_instruction + query, hits=hits)
        passages = self.corpus[indices[0]]
        doc_list = []
        for i in range(hits):
            s = passages['title'][i] + ' -- ' + passages['text'][i]
            doc_list.append(s)
        return doc_list