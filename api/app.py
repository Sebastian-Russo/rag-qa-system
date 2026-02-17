"""
Harry Potter RAG Q&A System - Flask API
Routes only — all logic lives in src/
"""
import sys
import os

# Add project root to path so we can import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from src.retriever import Retriever
from src.generator import generate_answer

app = Flask(__name__)

# Initialize retriever once on startup
retriever = Retriever()
print("✓ API ready!")


@app.route('/ask', methods=['POST'])
def ask():
    """
    Full RAG pipeline: retrieve + generate
    Body: {
        "question": "How did Harry get his scar?",
        "top_k": 10,
        "use_expansion": true,
        "use_reranker": true
    }
    """
    data = request.get_json()

    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    question = data['question']
    top_k = data.get('top_k', 10)
    use_expansion = data.get('use_expansion', True)
    use_reranker = data.get('use_reranker', True)

    # Retrieve
    results, queries = retriever.search(
        question, top_k=top_k,
        use_expansion=use_expansion,
        use_reranker=use_reranker
    )

    # Generate
    answer = generate_answer(question, results)

    return jsonify({
        "question": question,
        "answer": answer,
        "expanded_queries": queries,
        "sources": [
            {
                "chunk_id": r["chunk_id"],
                "preview": r["text"][:200],
                "semantic_score": r["semantic_score"],
                "keyword_score": r["keyword_score"],
                "rerank_score": r.get("rerank_score")
            }
            for r in results
        ],
        "settings": {
            "top_k": top_k,
            "use_expansion": use_expansion,
            "use_reranker": use_reranker
        }
    })


@app.route('/search', methods=['POST'])
def search():
    """
    Retrieval only — no answer generation
    Useful for debugging what chunks are found
    Body: {"question": "Dobby", "top_k": 5}
    """
    data = request.get_json()

    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    question = data['question']
    top_k = data.get('top_k', 5)

    results, queries = retriever.search(question, top_k=top_k)

    return jsonify({
        "question": question,
        "expanded_queries": queries,
        "results": [
            {
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "semantic_score": r["semantic_score"],
                "keyword_score": r["keyword_score"],
                "rerank_score": r.get("rerank_score")
            }
            for r in results
        ]
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "chunks_loaded": len(retriever.chunks),
        "embedding_dim": retriever.embeddings.shape[1]
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
