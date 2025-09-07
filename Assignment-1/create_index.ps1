python -m pyserini.index.lucene `
  --input wiki_jsonl `
  --index wiki_index `
  --generator DefaultLuceneDocumentGenerator `
  --collection JsonCollection `
  --language en
