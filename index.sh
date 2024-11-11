python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input dataset/hotpot/corpus \
  --index /data/webgpt/IIA/dataset/hotpot/index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw