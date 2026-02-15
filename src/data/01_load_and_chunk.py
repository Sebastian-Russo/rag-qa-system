"""
Phase 1: Load PDFs and chunk text for RAG
Extracts text from Harry Potter PDFs, splits into chunks with overlap
"""
import os
import pickle
from pathlib import Path

# pip install pymupdf
import fitz  # PyMuPDF

print("=" * 60)
print("PHASE 1: LOAD AND CHUNK DOCUMENTS")
print("=" * 60)

# ============================================================
# EXTRACT TEXT FROM PDFs
# ============================================================
print("\n" + "=" * 60)
print("EXTRACTING TEXT FROM PDFs")
print("=" * 60)

raw_dir = Path("data/raw")
documents = []

# Start with the main collection
target_files = [
    "Harry Potter: The Complete Collection (1-7).pdf"
]

for filename in target_files:
    filepath = raw_dir / filename
    print(f"\nProcessing: {filename}")
    print(f"  Size: {filepath.stat().st_size / 1024 / 1024:.1f} MB")

    doc = fitz.open(str(filepath))
    print(f"  Pages: {len(doc)}")

    full_text = ""
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            full_text += text + "\n"

        if (page_num + 1) % 100 == 0:
            print(f"  Extracted {page_num + 1}/{len(doc)} pages...")

    documents.append({
        "source": filename,
        "text": full_text,
        "pages": len(doc)
    })
    doc.close()

    print(f"  ✓ Extracted {len(full_text):,} characters")

# ============================================================
# CHUNK TEXT
# ============================================================
print("\n" + "=" * 60)
print("CHUNKING TEXT")
print("=" * 60)

"""
WHY CHUNK?
- LLMs have context limits (can't feed entire book at once)
- Embeddings work better on focused passages
- Retrieval is more precise with smaller chunks

CHUNK SIZE TRADE-OFF:
- Too small (100 chars): Loses context, fragments sentences
- Too big (5000 chars): Too vague for precise retrieval
- Sweet spot: 500-1000 chars with overlap

OVERLAP:
- Without overlap: Information at chunk boundaries gets lost
- With overlap: Same sentence appears in adjacent chunks
- Ensures no context falls through the cracks
"""

CHUNK_SIZE = 800       # Characters per chunk
CHUNK_OVERLAP = 200    # Overlap between consecutive chunks

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping chunks

    Example with chunk_size=10, overlap=3:
    Text: "The wizard cast a powerful spell on the dragon"
    Chunk 1: "The wizard"
    Chunk 2: "ard cast a"    (overlaps "ard" from chunk 1)
    Chunk 3: "t a powerf"   (overlaps "t a" from chunk 2)
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence boundary
        chunk = text[start:end]

        # Look for last period, question mark, or newline near the end
        for sep in ['. ', '? ', '! ', '\n']:
            last_sep = chunk.rfind(sep)
            if last_sep > chunk_size * 0.5:  # Only if it's past halfway
                chunk = chunk[:last_sep + 1]
                end = start + last_sep + 1
                break

        if chunk.strip():
            chunks.append({
                "text": chunk.strip(),
                "start_char": start,
                "chunk_id": len(chunks)
            })

        start = end - overlap

    return chunks


all_chunks = []

for doc in documents:
    print(f"\nChunking: {doc['source']}")
    chunks = chunk_text(doc['text'])

    for chunk in chunks:
        chunk['source'] = doc['source']

    all_chunks.extend(chunks)
    print(f"  ✓ Created {len(chunks)} chunks")

print(f"\n{'=' * 60}")
print(f"TOTAL CHUNKS: {len(all_chunks)}")
print(f"{'=' * 60}")

# Show sample chunks
print("\n--- Sample Chunks ---")
for i in [0, len(all_chunks) // 2, len(all_chunks) - 1]:
    print(f"\nChunk {i}:")
    print(f"  Source: {all_chunks[i]['source']}")
    print(f"  Length: {len(all_chunks[i]['text'])} chars")
    print(f"  Preview: {all_chunks[i]['text'][:150]}...")

# ============================================================
# SAVE CHUNKS
# ============================================================
print("\n" + "=" * 60)
print("SAVING CHUNKS")
print("=" * 60)

output_path = Path("data/processed/chunks.pkl")
with open(output_path, 'wb') as f:
    pickle.dump(all_chunks, f)

print(f"✓ Saved {len(all_chunks)} chunks to {output_path}")
print(f"✓ Phase 1 complete!")
