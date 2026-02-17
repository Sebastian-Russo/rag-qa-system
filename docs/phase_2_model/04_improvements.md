# Improvements

There are a few approaches we can stack together. Let me explain each one, then we'll implement them.

**1. Hybrid Search (Keyword + Semantic)** — Right now we only do semantic search (meaning-based). But sometimes you want exact word matching. If you ask "Who is Dobby?", semantic search might find chunks about house-elves in general. Keyword search would find every chunk that literally contains "Dobby." Combining both gives better results.

**2. Re-ranking** — Right now we grab the top 10 closest chunks and send all of them to Claude. But some of those chunks might be noise. Re-ranking takes the initial results, scores them more carefully against the question, and keeps only the best ones. Think of it as a second pass — the first search is fast and broad, the re-ranker is slower but more precise.

**3. Query Expansion** — When you asked "how old is Harry in the first book" it failed because the books never phrase it that way. Query expansion takes your question and generates alternative phrasings: "Harry's age", "Harry's eleventh birthday", "Harry turned eleven." Then it searches for all of them and merges the results.

## Output

$ python3 src/models/04_retrieval_improvements.py
============================================================
HARRY POTTER RAG Q&A SYSTEM
============================================================

Loading vector store...
✓ Loaded 12921 chunks with 384-dim vectors

Loading embedding model...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|██| 103/103 [00:00<00:00, 4165.03it/s, Materializing param=pooler.dense.weight]
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  |
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  |

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
✓ Embedding model loaded
Loading re-ranker model...
config.json: 100%|██████████████████████████████████████████████████| 794/794 [00:00<00:00, 5.37MB/s]
model.safetensors: 100%|████████████████████████████████████████| 90.9M/90.9M [00:02<00:00, 41.0MB/s]
Loading weights: 100%|████| 105/105 [00:00<00:00, 4027.56it/s, Materializing param=classifier.weight]
BertForSequenceClassification LOAD REPORT from: cross-encoder/ms-marco-MiniLM-L-6-v2
Key                          | Status     |  |
-----------------------------+------------+--+-
bert.embeddings.position_ids | UNEXPECTED |  |

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
tokenizer_config.json: 1.33kB [00:00, 3.89MB/s]
vocab.txt: 232kB [00:00, 6.53MB/s]
tokenizer.json: 711kB [00:00, 35.9MB/s]
special_tokens_map.json: 100%|███████████████████████████████████████| 132/132 [00:00<00:00, 956kB/s]
README.md: 3.67kB [00:00, 8.71MB/s]
✓ Re-ranker loaded

============================================================
INTERACTIVE MODE
============================================================

Ask any Harry Potter question!
Commands:
  'quit'       - exit
  'top X'      - change number of chunks retrieved (default 10)
  'sources'    - toggle showing retrieved passages
  'expansion'  - toggle query expansion on/off
  'reranker'   - toggle re-ranking on/off
------------------------------------------------------------

📖 Ask: what spell made a deer for harry

Question: what spell made a deer for harry
------------------------------------------------------------
Retrieving relevant passages...
  Expanded into 4 queries:
    - what spell made a deer for harry
    - Expecto Patronum conjured a silver deer
    - Harry's patronus took the shape of a stag
    - A silvery stag emerged from Harry's wand

Top 10 passages found:

  [1] semantic: 0.4694, keyword: 1.0000, rerank: -1.3856
      was almost as though an invisible
beam of understanding shot between them.
“The Reverse Spell effect?” said Sirius sharply.
“Exactly,” said Dumbledore...

  [2] semantic: 0.5178, keyword: 0.8333, rerank: -3.3112
      perio!”

A curious sensation shot down Harry’s arm, a feeling of tingling warmth
that seemed to flow from his mind, down the sinews and veins connecti...

  [3] semantic: 0.5782, keyword: 1.0000, rerank: -4.6551
      . . . .
Four summers ago, on his eleventh birthday, he had entered Mr.
Ollivander’s shop with Hagrid to buy a wand. Mr. Ollivander had taken his
measu...

  [4] semantic: 0.5067, keyword: 0.8333, rerank: -4.7723
      ad
splintered apart completely. Harry took it into his hands as though it was a
living thing that had suffered a terrible injury. He could not think p...

  [5] semantic: 0.5878, keyword: 0.6667, rerank: -4.8361
      ermione’s, and squinting up the path. Harry dug in
the pockets of his jacket for his own wand — but it wasn’t there. The only
thing he could find was ...

  [6] semantic: 0.6242, keyword: 0.5833, rerank: -5.1141
      he turned a corner, he saw . . . a dementor gliding toward him.
Twelve feet tall, its face hidden by its hood, its rotting, scabbed hands
outstretched...

  [7] semantic: 0.5126, keyword: 0.8333, rerank: -5.1691
      ,” said the man. “Yes, yes. I thought I’d be seeing you soon.
Harry Potter.” It wasn’t a question. “You have your mother’s eyes. It seems
only yesterd...

  [8] semantic: 0.5013, keyword: 0.8333, rerank: -5.2714
      n asked for identification before!”
said Hermione.
“They know!” whispered Griphook in Harry’s ear. “They must have been
warned there might be an impos...

  [9] semantic: 0.5435, keyword: 0.7500, rerank: -5.2902
      ng your
wands to ensure that they are in good condition before the tournament.”
Harry looked around, and with a jolt of surprise saw an old wizard wit...

  [10] semantic: 0.5634, keyword: 0.7097, rerank: -5.4178
      ered, “we’re still fighting. Come on,
now. . . .”
There was a silver spark, then a wavering light, and then, with the
greatest effort it had ever cost...

Generating answer...

Answer:
Based on the provided passages, the spell that made a deer (specifically a silver stag) for Harry was **"Expecto Patronum"**.

This is clearly stated in Passage 6: "He summoned the happiest thought he could, concentrated with all his might on the thought of getting out of the maze and celebrating with Ron and Hermione, raised his wand, and cried, 'Expecto Patronum!' A silver stag erupted from the end of Harry's wand and galloped toward the dementor..."

This is confirmed again in Passage 10: "There was a silver spark, then a wavering light, and then, with the greatest effort it had ever cost him, the stag burst from the end of Harry's wand."

The Expecto Patronum spell produces Harry's Patronus, which takes the form of a silver stag (a male deer).

============================================================

📖 Ask:
