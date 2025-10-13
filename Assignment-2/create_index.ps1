python -m pyserini.index.lucene `
  --input data/docs `
  --index hotpotqa_index `
  --generator DefaultLuceneDocumentGenerator `
  --collection JsonCollection `
  --language en `
  --threads 4 `
  --storePositions --storeDocvectors --storeRaw