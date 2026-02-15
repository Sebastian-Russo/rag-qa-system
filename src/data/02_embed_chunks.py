"""
Phase 2: Embed chunks into vectors for semantic search
Converts text chunks into numerical vectors using a sentence transformer model

HOW EMBEDDINGS WORK:
- Each chunk gets converted to a vector (list of 384 numbers)
- Similar meaning = vectors point in similar direction
- "Harry cast a spell" and "Potter used magic" → close together
- "Harry cast a spell" and "Dumbledore ate dinner" → far apart
- This lets us search by MEANING, not just keyword matching
"""
import pickle
import numpy as np
from pathlib import Path

# pip install sentence-transformers
from sentence_transformers import SentenceTransformer

print("=" * 60)
print("PHASE 2: EMBED CHUNKS")
print("=" * 60)

# ============================================================
# LOAD CHUNKS
# ============================================================
print("\nLoading chunks...")
with open("data/processed/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
print(f"✓ Loaded {len(chunks)} chunks")

# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================
print("\n" + "=" * 60)
print("LOADING EMBEDDING MODEL")
print("=" * 60)

"""
MODEL: all-MiniLM-L6-v2
- Small and fast (80MB)
- Produces 384-dimensional vectors
- Great quality for its size
- Trained on 1 billion+ sentence pairs

WHY THIS MODEL:
- Bigger models (e.g., all-mpnet-base-v2) are more accurate
  but slower and heavier
- For our use case, MiniLM is the sweet spot
- Same model used in most RAG tutorials and production systems
"""
print("\nLoading sentence-transformers model...")
print("(Downloads ~80MB on first run)")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✓ Model loaded")

# Quick demo of what embeddings look like
demo_sentences = [
    "Harry Potter cast a spell",
    "Potter used his wand for magic",
    "Dumbledore enjoyed lemon drops"
]
demo_embeddings = model.encode(demo_sentences)

print("\n--- Embedding Demo ---")
print(f"Sentence: '{demo_sentences[0]}'")
print(f"Vector shape: {demo_embeddings[0].shape}")
print(f"First 10 values: {demo_embeddings[0][:10].round(4)}")

# Show similarity
from numpy.linalg import norm
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

sim_related = cosine_similarity(demo_embeddings[0], demo_embeddings[1])
sim_unrelated = cosine_similarity(demo_embeddings[0], demo_embeddings[2])
print(f"\nSimilarity ('cast a spell' vs 'used magic'): {sim_related:.4f}  ← RELATED")
print(f"Similarity ('cast a spell' vs 'lemon drops'): {sim_unrelated:.4f}  ← UNRELATED")

# ============================================================
# EMBED ALL CHUNKS
# ============================================================
print("\n" + "=" * 60)
print("EMBEDDING ALL CHUNKS")
print("=" * 60)

texts = [chunk['text'] for chunk in chunks]

print(f"\nEmbedding {len(texts)} chunks...")
print("This may take a few minutes on CPU...\n")

# Encode in batches (shows progress)
BATCH_SIZE = 256
all_embeddings = []

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i:i + BATCH_SIZE]
    batch_embeddings = model.encode(batch, show_progress_bar=False)
    all_embeddings.append(batch_embeddings)

    done = min(i + BATCH_SIZE, len(texts))
    pct = done / len(texts) * 100
    print(f"  Embedded {done}/{len(texts)} chunks ({pct:.0f}%)")

embeddings = np.vstack(all_embeddings)
print(f"\n✓ All chunks embedded")
print(f"  Shape: {embeddings.shape}  ({embeddings.shape[0]} chunks × {embeddings.shape[1]} dimensions)")

# ============================================================
# SAVE EMBEDDINGS
# ============================================================
print("\n" + "=" * 60)
print("SAVING EMBEDDINGS")
print("=" * 60)

output_path = Path("data/vectorstore")

np.save(output_path / "embeddings.npy", embeddings)
print(f"✓ Saved embeddings: {output_path / 'embeddings.npy'}")

with open(output_path / "chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)
print(f"✓ Saved chunks: {output_path / 'chunks.pkl'}")

print(f"\n✓ Phase 2 complete!")
print(f"  {len(chunks)} chunks embedded into {embeddings.shape[1]}-dimensional vectors")
print(f"  Ready for semantic search!")
