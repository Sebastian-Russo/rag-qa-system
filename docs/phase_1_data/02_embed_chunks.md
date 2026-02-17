
This will take a few minutes to embed all 12,921 chunks. I want to see two things from the output: the similarity demo (should show the related sentences scoring higher) and confirmation that all chunks embedded successfully.

---------------------------------------------------------------------------------------
## Output

$ python3 src/models/02_embed_chunks.py

## PHASE 2: EMBED CHUNKS

Loading chunks...
✓ Loaded 12921 chunks

## LOAD EMBEDDING MODEL

Loading sentence-transformers model...
(Downloads ~80MB on first run)
modules.json: 100%|█████████████████████████████████████████████████| 349/349 [00:00<00:00, 1.59MB/s]
config_sentence_transformers.json: 100%|█████████████████████████████| 116/116 [00:00<00:00, 999kB/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
README.md: 10.5kB [00:00, 45.6MB/s]
sentence_bert_config.json: 100%|███████████████████████████████████| 53.0/53.0 [00:00<00:00, 521kB/s]
config.json: 100%|██████████████████████████████████████████████████| 612/612 [00:00<00:00, 5.35MB/s]
model.safetensors: 100%|████████████████████████████████████████| 90.9M/90.9M [00:02<00:00, 34.4MB/s]
Loading weights: 100%|██| 103/103 [00:00<00:00, 3776.87it/s, Materializing param=pooler.dense.weight]
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  |
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  |

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
tokenizer_config.json: 100%|████████████████████████████████████████| 350/350 [00:00<00:00, 2.93MB/s]
vocab.txt: 232kB [00:00, 8.89MB/s]
tokenizer.json: 466kB [00:00, 24.0MB/s]
special_tokens_map.json: 100%|███████████████████████████████████████| 112/112 [00:00<00:00, 806kB/s]
config.json: 100%|██████████████████████████████████████████████████| 190/190 [00:00<00:00, 1.78MB/s]
✓ Model loaded

--- Embedding Demo ---
Sentence: 'Harry Potter cast a spell'
Vector shape: (384,)
First 10 values: [-0.0143  0.027  -0.0465  0.0036 -0.1313 -0.0278  0.0903  0.008   0.0132
 -0.0873]

Similarity ('cast a spell' vs 'used magic'): 0.6086  ← RELATED
Similarity ('cast a spell' vs 'lemon drops'): 0.2358  ← UNRELATED

## EMBEDDING ALL CHUNKS

Embedding 12921 chunks...
This may take a few minutes on CPU...

  Embedded 256/12921 chunks (2%)
  Embedded 512/12921 chunks (4%)
  Embedded 768/12921 chunks (6%)
  Embedded 1024/12921 chunks (8%)
  Embedded 1280/12921 chunks (10%)
  Embedded 1536/12921 chunks (12%)
  Embedded 1792/12921 chunks (14%)
  Embedded 2048/12921 chunks (16%)
  Embedded 2304/12921 chunks (18%)
  Embedded 2560/12921 chunks (20%)
  Embedded 2816/12921 chunks (22%)
  Embedded 3072/12921 chunks (24%)
  Embedded 3328/12921 chunks (26%)
  Embedded 3584/12921 chunks (28%)
  Embedded 3840/12921 chunks (30%)
  Embedded 4096/12921 chunks (32%)
  Embedded 4352/12921 chunks (34%)
  Embedded 4608/12921 chunks (36%)
  Embedded 4864/12921 chunks (38%)
  Embedded 5120/12921 chunks (40%)
  Embedded 5376/12921 chunks (42%)
  Embedded 5632/12921 chunks (44%)
  Embedded 5888/12921 chunks (46%)
  Embedded 6144/12921 chunks (48%)
  Embedded 6400/12921 chunks (50%)
  Embedded 6656/12921 chunks (52%)
  Embedded 6912/12921 chunks (53%)
  Embedded 7168/12921 chunks (55%)
  Embedded 7424/12921 chunks (57%)
  Embedded 7680/12921 chunks (59%)
  Embedded 7936/12921 chunks (61%)
  Embedded 8192/12921 chunks (63%)
  Embedded 8448/12921 chunks (65%)
  Embedded 8704/12921 chunks (67%)
  Embedded 8960/12921 chunks (69%)
  Embedded 9216/12921 chunks (71%)
  Embedded 9472/12921 chunks (73%)
  Embedded 9728/12921 chunks (75%)
  Embedded 9984/12921 chunks (77%)
  Embedded 10240/12921 chunks (79%)
  Embedded 10496/12921 chunks (81%)
  Embedded 10752/12921 chunks (83%)
  Embedded 11008/12921 chunks (85%)
  Embedded 11264/12921 chunks (87%)
  Embedded 11520/12921 chunks (89%)
  Embedded 11776/12921 chunks (91%)
  Embedded 12032/12921 chunks (93%)
  Embedded 12288/12921 chunks (95%)
  Embedded 12544/12921 chunks (97%)
  Embedded 12800/12921 chunks (99%)
  Embedded 12921/12921 chunks (100%)

✓ All chunks embedded
  Shape: (12921, 384)  (12921 chunks × 384 dimensions)

## SAVING EMBEDDINGS
✓ Saved embeddings: data/vectorstore/embeddings.npy
✓ Saved chunks: data/vectorstore/chunks.pkl

✓ Phase 2 complete!
  12921 chunks embedded into 384-dimensional vectors
  Ready for semantic search!

---------------------------------------------------------------------------------------

## Summary
