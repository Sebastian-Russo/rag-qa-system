"""
Phase 3: Retrieval and Answer Generation
Takes a question, finds relevant chunks, sends to LLM for answer

THE RAG PIPELINE:
1. User asks a question
2. Embed the question into a vector (same model as chunks)
3. Find most similar chunks using cosine similarity
4. Send chunks + question to LLM
5. LLM reads the chunks and answers the question
"""
import pickle
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
import requests
import json

print("=" * 60)
print("PHASE 3: RAG RETRIEVAL + GENERATION")
print("=" * 60)

# ============================================================
# LOAD EMBEDDINGS AND CHUNKS
# ============================================================
print("\nLoading vector store...")
embeddings = np.load("data/vectorstore/embeddings.npy")
with open("data/vectorstore/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
print(f"✓ Loaded {len(chunks)} chunks with {embeddings.shape[1]}-dim vectors")

# ============================================================
# LOAD EMBEDDING MODEL (same one used to embed chunks)
# ============================================================
print("\nLoading embedding model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✓ Embedding model loaded")

# ============================================================
# RETRIEVAL FUNCTION
# ============================================================
def retrieve(query, top_k=5):
    """
    Find the most relevant chunks for a query

    1. Embed the query into a vector
    2. Compare against all chunk vectors
    3. Return top_k most similar chunks

    COSINE SIMILARITY:
    - 1.0 = identical meaning
    - 0.0 = completely unrelated
    - Higher = more relevant
    """
    # Embed the question
    query_vector = embed_model.encode([query])[0]

    # Calculate similarity against all chunks
    similarities = np.dot(embeddings, query_vector) / (
        norm(embeddings, axis=1) * norm(query_vector)
    )

    # Get top_k most similar
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
ANTHROPIC_API_KEY = "YOUR_API_KEY_HERE"  # You'll add your key

def generate_answer(query, context_chunks):
    """
    Send retrieved chunks + question to Claude

    The system prompt restricts Claude to ONLY use
    the provided context — no outside knowledge
    """
    # Build context from retrieved chunks
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
        }
    )

    data = response.json()
    return data["content"][0]["text"]

# ============================================================
# RAG PIPELINE
# ============================================================
def ask(question, top_k=5, show_sources=True):
    """
    Full RAG pipeline: retrieve + generate
    """
    print(f"\nQuestion: {question}")
    print("-" * 60)

    # Step 1: Retrieve
    print("Retrieving relevant passages...")
    results = retrieve(question, top_k=top_k)

    if show_sources:
        print(f"\nTop {top_k} passages found:")
        for i, r in enumerate(results):
            print(f"\n  [{i+1}] Similarity: {r['similarity']:.4f}")
            print(f"      {r['text'][:150]}...")

    # Step 2: Generate
    print("\nGenerating answer...")
    answer = generate_answer(question, results)

    print(f"\nAnswer:\n{answer}")
    return answer

# ============================================================
# TEST IT
# ============================================================
print("\n" + "=" * 60)
print("TESTING RAG PIPELINE")
print("=" * 60)

test_questions = [
    "What spell did Harry use to defeat Voldemort?",
    "How did Harry get his scar?",
    "What are the three Deathly Hallows?",
    "Who is the Half-Blood Prince?",
]

for q in test_questions:
    ask(q)
    print("\n" + "=" * 60)
