python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input dataset/hotpot/corpus \
  --index /AssistRAG/dataset/hotpot/index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw
