python -m pyserini.index.lucene `
  --input data\bm25_docs `
  --index quora_trec_index `
  --generator DefaultLuceneDocumentGenerator `
  --collection JsonCollection `
  --language en `
  --threads 8 `
  --storePositions --storeDocvectors --storeRaw