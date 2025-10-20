python -m pyserini.index.lucene `
  --input data/docs_test `
  --index wikiclir_index `
  --generator DefaultLuceneDocumentGenerator `
  --collection JsonCollection `
  --language en `
  --threads 4 `
  --storePositions --storeDocvectors --storeRaw