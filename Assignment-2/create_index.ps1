python -m pyserini.index.lucene `
  --input data/docs `
  --index hotpotqa_index `
  --generator DefaultLuceneDocumentGenerator `
  --collection JsonCollection `
  --language en `
  --threads 8 `
  --storePositions --storeDocvectors --storeRaw

python -m pyserini.index.lucene `
  --input data/docs_test `
  --index wikiclir_index `
  --generator DefaultLuceneDocumentGenerator `
  --collection JsonCollection `
  --language en `
  --threads 8 `
  --storePositions --storeDocvectors --storeRaw