"""
Phase 3: Retrieval and Answer Generation (Interactive)
Takes a question, finds relevant chunks, sends to LLM for answer

THE RAG PIPELINE:
1. User asks a question
2. Embed the question into a vector (same model as chunks)
3. Find most similar chunks using cosine similarity
4. Send chunks + question to LLM
5. LLM reads the chunks and answers the question

ANALOGY - THE LIBRARY:
Imagine a massive library with 12,921 index cards (our chunks).
Each card has a book passage written on it.

- The EMBEDDING MODEL is a librarian who reads each card and assigns
  it a location in a giant 384-dimensional room. Cards about similar
  topics get placed near each other.

- When you ask a QUESTION, the librarian reads your question and
  figures out where in the room it belongs.

- RETRIEVAL is the librarian walking to that spot in the room and
  grabbing the 5 closest cards.

- GENERATION is handing those 5 cards to a reader (Claude) and
  saying "answer this question using ONLY what's on these cards."
"""
import pickle
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

print("=" * 60)
print("HARRY POTTER RAG Q&A SYSTEM")
print("=" * 60)

# ============================================================
# LOAD EMBEDDINGS AND CHUNKS
# ============================================================
# ANALOGY: Loading all 12,921 index cards and their locations
# in the 384-dimensional room. We prepared these in Phase 2.
print("\nLoading vector store...")
embeddings = np.load("data/vectorstore/embeddings.npy")
with open("data/vectorstore/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
print(f"✓ Loaded {len(chunks)} chunks with {embeddings.shape[1]}-dim vectors")

# ============================================================
# LOAD EMBEDDING MODEL (same one used to embed chunks)
# ============================================================
# ANALOGY: Hiring the same librarian who organized the cards.
# CRITICAL: Must be the same model! A different model would
# place things in different locations — like two librarians
# with different filing systems. Your question would end up
# in the wrong part of the room.
print("\nLoading embedding model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✓ Embedding model loaded")

# ============================================================
# RETRIEVAL FUNCTION
# ============================================================
def retrieve(query, top_k=5):
    """
    Find the most relevant chunks for a query

    ANALOGY: You hand your question to the librarian.
    The librarian reads it, walks to the right area of the room,
    and grabs the closest cards.

    COSINE SIMILARITY:
    - Measures the angle between two vectors
    - 1.0 = pointing same direction = identical meaning
    - 0.0 = perpendicular = completely unrelated
    - Think of it like compass directions:
      "Harry cast a spell" and "Potter used magic" point roughly north
      "Harry cast a spell" and "Dumbledore ate dinner" point in
      different directions entirely
    """
    # STEP 1: Convert your question into a vector
    # ANALOGY: The librarian reads your question and figures out
    # where in the room it belongs
    query_vector = embed_model.encode([query])[0]

    # STEP 2: Calculate similarity between your question and ALL chunks
    # ANALOGY: The librarian measures the distance from your question's
    # spot to every single card in the room (all 12,921 of them)
    similarities = np.dot(embeddings, query_vector) / (
        norm(embeddings, axis=1) * norm(query_vector)
    )

    # STEP 3: Grab the top_k most similar chunks
    # ANALOGY: The librarian picks up the 5 closest cards and
    # hands them to you, sorted by relevance
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "chunk_id": idx,
            "text": chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "similarity": float(similarities[idx])
        })

    return results

# ============================================================
# GENERATION FUNCTION (Claude API)
# ============================================================
def generate_answer(query, context_chunks):
    """
    Send retrieved chunks + question to Claude

    ANALOGY: You hand the 5 index cards to a reader (Claude) along
    with your question. You tell the reader:
    "Answer my question using ONLY what's written on these cards.
     If the answer isn't on the cards, say so."

    The reader (Claude) is smart — it can synthesize info across
    multiple cards, infer connections, and write a clean answer.
    But it's RESTRICTED to only use what's on the cards.

    This is the key difference from just asking Claude directly:
    - Normal Claude: Uses everything it learned during training
    - RAG Claude: Only uses the specific passages we retrieved
    """
    # Build context string from retrieved chunks
    # ANALOGY: Laying out the 5 index cards on the table
    context = "\n\n---\n\n".join([
        f"[Passage {i+1}]:\n{chunk['text']}"
        for i, chunk in enumerate(context_chunks)
    ])

    # Make API call with explicit timeout to prevent hanging
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
        timeout=30  # 30 second timeout to prevent indefinite hanging
    )

    data = response.json()

    # Error handling in case API call fails
    if "content" not in data:
        return f"API Error: {data.get('error', {}).get('message', 'Unknown error')}"

    return data["content"][0]["text"]

# ============================================================
# RAG PIPELINE
# ============================================================
def ask(question, top_k=5, show_sources=True):
    """
    Full RAG pipeline: retrieve + generate

    ANALOGY - FULL FLOW:
    1. You walk into the library and ask a question
    2. Librarian finds the 5 most relevant index cards
    3. Librarian hands the cards to the reader (Claude)
    4. Reader studies the cards and writes you an answer
    5. You get the answer + can see which cards were used
    """
    print(f"\nQuestion: {question}")
    print("-" * 60)

    # Step 1: Retrieve
    # ANALOGY: Librarian searches the room for relevant cards
    print("Retrieving relevant passages...")
    results = retrieve(question, top_k=top_k)

    if show_sources:
        print(f"\nTop {top_k} passages found:")
        for i, r in enumerate(results):
            print(f"\n  [{i+1}] Similarity: {r['similarity']:.4f}")
            print(f"      {r['text'][:150]}...")

    # Step 2: Generate
    # ANALOGY: Reader (Claude) studies the cards and answers
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
print("  'quit'     - exit")
print("  'top X'    - change number of chunks retrieved (default 5)")
print("  'sources'  - toggle showing retrieved passages")
print("-" * 60)

top_k = 5
show_sources = True

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

    ask(question, top_k=top_k, show_sources=show_sources)
    print("\n" + "=" * 60)
