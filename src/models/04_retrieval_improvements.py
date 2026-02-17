"""
Phase 3: Retrieval and Answer Generation (Interactive)
Takes a question, finds relevant chunks, sends to LLM for answer

THE RAG PIPELINE:
1. User asks a question
2. Expand the question into multiple phrasings
3. Embed each phrasing and search by meaning (semantic)
4. Also search by exact words (keyword)
5. Combine and re-rank results
6. Send best chunks + question to LLM
7. LLM reads the chunks and answers the question

ANALOGY - THE UPGRADED LIBRARY:
Before, we had one librarian doing one search. Now we have a team:

1. QUERY EXPANSION - A translator who rephrases your question
   multiple ways before searching. You ask "How old is Harry in
   the first book?" and the translator also writes:
   "Harry's age", "Harry's eleventh birthday", "Harry turned eleven"
   Now the librarian searches for ALL of those.

2. HYBRID SEARCH - Two search methods working together:
   - The LIBRARIAN searches by meaning (semantic)
   - A CTRL+F ROBOT searches by exact words (keyword)
   Their scores get combined: 70% librarian, 30% robot.

3. RE-RANKER - A senior librarian who reviews the results.
   The first search is fast but rough. The re-ranker reads each
   result more carefully and asks "does this ACTUALLY answer the
   question?" Bad results get filtered out. Good ones get boosted.
"""
import pickle
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer, CrossEncoder
import requests
import json
import os
import re
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Timeout configuration (in seconds)
REQUEST_TIMEOUT = (10, 30)  # (connect_timeout, read_timeout)

print("=" * 60)
print("HARRY POTTER RAG Q&A SYSTEM")
print("=" * 60)

# ============================================================
# LOAD EMBEDDINGS AND CHUNKS
# ============================================================
# ANALOGY: Loading all 12,921 index cards and their locations
# in the 384-dimensional room.
print("\nLoading vector store...")
embeddings = np.load("data/vectorstore/embeddings.npy")
with open("data/vectorstore/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
print(f"✓ Loaded {len(chunks)} chunks with {embeddings.shape[1]}-dim vectors")

# ============================================================
# LOAD MODELS
# ============================================================
# ANALOGY: Hiring our team of librarians.

# The embedding model (same one that organized the cards)
print("\nLoading embedding model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✓ Embedding model loaded")

# The re-ranker (senior librarian who double-checks results)
# This is a CrossEncoder — it takes a (question, passage) PAIR
# and scores how relevant the passage is to the question.
# It's slower but much more accurate than cosine similarity.
#
# ANALOGY: The first search is like scanning book covers.
# The re-ranker actually READS each passage and asks
# "does this answer the question?" Much more careful.
print("Loading re-ranker model...")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("✓ Re-ranker loaded")

# ============================================================
# QUERY EXPANSION
# ============================================================
def expand_query(query):
    """
    Generate alternative phrasings of the question

    ANALOGY: You ask "How old is Harry in the first book?"
    The translator rewrites this as:
      - "Harry Potter age first book"  (keyword-friendly)
      - "Harry's age Sorcerer's Stone" (book-specific)
      - "Harry turned eleven"          (how the book phrases it)

    WHY THIS HELPS:
    The books never say "Harry is 11 in the first book."
    They say "on his eleventh birthday" or "he was turning eleven."
    If we only search with your exact wording, we miss those chunks.
    By searching multiple phrasings, we cast a wider net.

    We use Claude to generate the expansions because it understands
    how information is typically phrased in books vs questions.
    """
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        },
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 256,
            "system": """Generate 3 alternative phrasings of the user's question
for searching through Harry Potter book text. Think about how the
information would actually be written in the books, not how someone
would ask about it. Return ONLY the 3 phrasings, one per line, no
numbering or extra text.""",
            "messages": [
                {"role": "user", "content": query}
            ]
        },
        timeout=REQUEST_TIMEOUT
    )

    data = response.json()
    if "content" not in data:
        return [query]

    expanded = data["content"][0]["text"].strip().split("\n")
    expanded = [q.strip() for q in expanded if q.strip()]

    # Always include the original query
    return [query] + expanded


# ============================================================
# KEYWORD SEARCH
# ============================================================
def keyword_search(query):
    """
    Search by exact word matching across all chunks

    ANALOGY: The Ctrl+F robot. It doesn't understand meaning
    at all — it just counts how many of your search words
    appear in each chunk. Simple but effective for names,
    spells, and specific terms.

    "Who is Dobby?" → finds every chunk containing "Dobby"
    even if semantic search ranked those chunks lower.
    """
    # Extract meaningful words (ignore short ones like "is", "a", "the")
    query_terms = [
        term.lower() for term in re.findall(r'\w+', query)
        if len(term) > 2
    ]

    scores = []
    for chunk in chunks:
        text_lower = chunk['text'].lower()
        score = 0
        for term in query_terms:
            score += text_lower.count(term)
        scores.append(score)

    scores = np.array(scores, dtype=float)

    # Normalize to 0-1 range
    max_score = scores.max()
    if max_score > 0:
        scores = scores / max_score

    return scores


# ============================================================
# HYBRID RETRIEVAL
# ============================================================
def retrieve(query, top_k=10, semantic_weight=0.7, keyword_weight=0.3,
             use_expansion=True, use_reranker=True):
    """
    Full retrieval pipeline: expansion → hybrid search → re-rank

    ANALOGY - THE FULL PROCESS:

    Step 1 (Query Expansion):
      Your question gets rephrased 3 extra ways.
      Now we have 4 questions to search with.

    Step 2 (Semantic Search):
      The librarian searches the 384-dimensional room
      for each of the 4 questions. Results get merged.
      Chunks found by multiple phrasings score higher.

    Step 3 (Keyword Search):
      The Ctrl+F robot scans all chunks for exact word matches
      across all 4 phrasings.

    Step 4 (Combine):
      Semantic scores (70%) + keyword scores (30%) = combined score.
      This balances meaning with exact matching.

    Step 5 (Re-rank):
      The senior librarian takes the top candidates and reads each one
      carefully alongside the original question. Irrelevant results
      get dropped. The best ones rise to the top.

    Result: Much better retrieval than any single method alone.
    """
    # STEP 1: Query Expansion
    # ANALOGY: The translator generates alternative phrasings
    if use_expansion:
        queries = expand_query(query)
        print(f"  Expanded into {len(queries)} queries:")
        for q in queries:
            print(f"    - {q}")
    else:
        queries = [query]

    # STEP 2: Semantic Search (across all query phrasings)
    # ANALOGY: The librarian searches for each phrasing and
    # combines the results. A chunk that matches multiple
    # phrasings gets a higher score.
    semantic_scores = np.zeros(len(chunks))
    for q in queries:
        query_vector = embed_model.encode([q])[0]
        scores = np.dot(embeddings, query_vector) / (
            norm(embeddings, axis=1) * norm(query_vector)
        )
        semantic_scores = np.maximum(semantic_scores, scores)

    # STEP 3: Keyword Search (across all query phrasings)
    # ANALOGY: Ctrl+F robot searches for words from all phrasings
    kw_scores = np.zeros(len(chunks))
    for q in queries:
        scores = keyword_search(q)
        kw_scores = np.maximum(kw_scores, scores)

    # STEP 4: Combine Scores
    # ANALOGY: Merging the librarian's and robot's rankings
    combined_scores = (semantic_weight * semantic_scores) + (keyword_weight * kw_scores)

    # Get initial candidates (grab more than we need for re-ranking)
    n_candidates = top_k * 3 if use_reranker else top_k
    top_indices = np.argsort(combined_scores)[::-1][:n_candidates]

    candidates = []
    for idx in top_indices:
        candidates.append({
            "chunk_id": int(idx),
            "text": chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "combined_score": float(combined_scores[idx]),
            "semantic_score": float(semantic_scores[idx]),
            "keyword_score": float(kw_scores[idx])
        })

    # STEP 5: Re-rank
    # ANALOGY: The senior librarian reads each candidate passage
    # alongside the original question and scores how relevant it
    # actually is. This catches cases where a chunk matched on
    # surface-level words but doesn't actually answer the question.
    #
    # HOW IT WORKS:
    # CrossEncoder takes (question, passage) as a PAIR and outputs
    # a single relevance score. Unlike cosine similarity which
    # compares vectors independently, this model reads both texts
    # together and understands their relationship.
    if use_reranker:
        pairs = [[query, c["text"]] for c in candidates]
        rerank_scores = reranker.predict(pairs)

        for i, score in enumerate(rerank_scores):
            candidates[i]["rerank_score"] = float(score)

        # Sort by re-rank score
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    # Return top_k results
    return candidates[:top_k]


# ============================================================
# GENERATION FUNCTION (Claude API)
# ============================================================
def generate_answer(query, context_chunks):
    """
    Send retrieved chunks + question to Claude

    ANALOGY: Hand the best index cards to the reader (Claude).
    The reader studies them and writes an answer using ONLY
    what's on those cards — no outside knowledge.
    """
    context = "\n\n---\n\n".join([
        f"[Passage {i+1}]:\n{chunk['text']}"
        for i, chunk in enumerate(context_chunks)
    ])

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        },
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": """You are a Harry Potter expert assistant.
Answer questions ONLY using the provided context passages.
If the answer is not in the context, say "I couldn't find that in the provided passages."
Be specific and cite which part of the text supports your answer.""",
            "messages": [
                {
                    "role": "user",
                    "content": f"""Context passages from Harry Potter:

{context}

---

Question: {query}

Answer based ONLY on the passages above."""
                }
            ]
        },
        timeout=REQUEST_TIMEOUT
    )

    data = response.json()
    if "content" not in data:
        return f"API Error: {data.get('error', {}).get('message', 'Unknown error')}"

    return data["content"][0]["text"]


# ============================================================
# RAG PIPELINE
# ============================================================
def ask(question, top_k=10, show_sources=True):
    """
    Full RAG pipeline: expand → retrieve → re-rank → generate

    ANALOGY - THE COMPLETE FLOW:
    1. You walk into the library and ask a question
    2. The translator rephrases your question multiple ways
    3. The librarian (semantic) and Ctrl+F robot (keyword) both search
    4. Their results get combined (70/30 split)
    5. The senior librarian re-ranks the results
    6. The best passages go to the reader (Claude)
    7. Claude reads them and writes your answer
    """
    print(f"\nQuestion: {question}")
    print("-" * 60)

    # Retrieve with all improvements
    print("Retrieving relevant passages...")
    results = retrieve(question, top_k=top_k)

    if show_sources:
        print(f"\nTop {top_k} passages found:")
        for i, r in enumerate(results):
            rerank = f", rerank: {r['rerank_score']:.4f}" if 'rerank_score' in r else ""
            print(f"\n  [{i+1}] semantic: {r['semantic_score']:.4f}, "
                  f"keyword: {r['keyword_score']:.4f}{rerank}")
            print(f"      {r['text'][:150]}...")

    # Generate answer
    print("\nGenerating answer...")
    answer = generate_answer(question, results)

    print(f"\nAnswer:\n{answer}")
    return answer


# ============================================================
# INTERACTIVE MODE
# ============================================================
print("\n" + "=" * 60)
print("INTERACTIVE MODE")
print("=" * 60)
print("\nAsk any Harry Potter question!")
print("Commands:")
print("  'quit'       - exit")
print("  'top X'      - change number of chunks retrieved (default 10)")
print("  'sources'    - toggle showing retrieved passages")
print("  'expansion'  - toggle query expansion on/off")
print("  'reranker'   - toggle re-ranking on/off")
print("-" * 60)

top_k = 10
show_sources = True
use_expansion = True
use_reranker = True

while True:
    question = input("\n📖 Ask: ").strip()

    if not question:
        continue
    if question.lower() == 'quit':
        print("Goodbye!")
        break
    if question.lower().startswith('top '):
        top_k = int(question.split()[1])
        print(f"✓ Now retrieving top {top_k} chunks")
        continue
    if question.lower() == 'sources':
        show_sources = not show_sources
        print(f"✓ Show sources: {'ON' if show_sources else 'OFF'}")
        continue
    if question.lower() == 'expansion':
        use_expansion = not use_expansion
        print(f"✓ Query expansion: {'ON' if use_expansion else 'OFF'}")
        continue
    if question.lower() == 'reranker':
        use_reranker = not use_reranker
        print(f"✓ Re-ranker: {'ON' if use_reranker else 'OFF'}")
        continue

    ask(question, top_k=top_k, show_sources=show_sources)
    print("\n" + "=" * 60)
