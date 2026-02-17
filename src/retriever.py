"""
Retriever
Handles all search logic: semantic, keyword, hybrid, and re-ranking

ANALOGY: This is the library's search system.
- Semantic search = librarian who understands meaning
- Keyword search = Ctrl+F robot matching exact words
- Hybrid = combining both (70/30 split)
- Re-ranker = senior librarian who double-checks results
"""
import pickle
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer, CrossEncoder
import re
from src.config import (
    EMBEDDING_MODEL, RERANKER_MODEL, EMBEDDINGS_PATH, CHUNKS_PATH,
    DEFAULT_TOP_K, SEMANTIC_WEIGHT, KEYWORD_WEIGHT
)
from src.query_expander import expand_query


class Retriever:
    """
    Loads vector store and models once, reuses for every query.

    WHY A CLASS?
    Loading embeddings (12,921 x 384 matrix) and models takes time.
    We load once when the class is created, then every call to
    search() reuses them. Without a class, we'd reload everything
    on every question.
    """

    def __init__(self):
        """Load all data and models on initialization"""
        print("Loading vector store...")
        self.embeddings = np.load(EMBEDDINGS_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)
        print(f"✓ Loaded {len(self.chunks)} chunks with {self.embeddings.shape[1]}-dim vectors")

        print("Loading embedding model...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        print("✓ Embedding model loaded")

        print("Loading re-ranker...")
        self.reranker = CrossEncoder(RERANKER_MODEL)
        print("✓ Re-ranker loaded")

    def _semantic_search(self, queries):
        """
        Search by meaning across all chunks

        ANALOGY: The librarian reads each query phrasing, walks to
        the right area of the 384-dimensional room, and scores
        every card by distance. A chunk that matches multiple
        phrasings gets the highest score from any of them.
        """
        scores = np.zeros(len(self.chunks))
        for q in queries:
            query_vector = self.embed_model.encode([q])[0]
            similarities = np.dot(self.embeddings, query_vector) / (
                norm(self.embeddings, axis=1) * norm(query_vector)
            )
            scores = np.maximum(scores, similarities)
        return scores

    def _keyword_search(self, queries):
        """
        Search by exact word matching across all chunks

        ANALOGY: Ctrl+F robot scans every chunk for exact words.
        Good for names (Dobby), spells (Expelliarmus), and
        specific terms that semantic search might miss.
        """
        all_scores = np.zeros(len(self.chunks))

        for query in queries:
            query_terms = [
                term.lower() for term in re.findall(r'\w+', query)
                if len(term) > 2
            ]

            scores = []
            for chunk in self.chunks:
                text_lower = chunk['text'].lower()
                score = sum(text_lower.count(term) for term in query_terms)
                scores.append(score)

            scores = np.array(scores, dtype=float)
            max_score = scores.max()
            if max_score > 0:
                scores = scores / max_score

            all_scores = np.maximum(all_scores, scores)

        return all_scores

    def _rerank(self, query, candidates):
        """
        Re-rank candidates using CrossEncoder

        ANALOGY: The senior librarian reads each candidate passage
        alongside the question and asks "does this ACTUALLY answer
        the question?" Much more careful than the initial fast search.

        CrossEncoder reads (question, passage) as a PAIR — unlike
        cosine similarity which compares vectors independently.
        """
        pairs = [[query, c["text"]] for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        for i, score in enumerate(rerank_scores):
            candidates[i]["rerank_score"] = float(score)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates

    def search(self, query, top_k=DEFAULT_TOP_K,
               use_expansion=True, use_reranker=True):
        """
        Full retrieval pipeline: expand → hybrid search → re-rank

        ANALOGY - FULL PROCESS:
        1. Translator rephrases question (expansion)
        2. Librarian searches by meaning (semantic)
        3. Robot searches by exact words (keyword)
        4. Scores combined 70/30
        5. Senior librarian re-checks top results (re-rank)
        """
        # Step 1: Expand query
        if use_expansion:
            queries = expand_query(query)
        else:
            queries = [query]

        # Step 2: Semantic search
        semantic_scores = self._semantic_search(queries)

        # Step 3: Keyword search
        kw_scores = self._keyword_search(queries)

        # Step 4: Combine scores
        combined = (SEMANTIC_WEIGHT * semantic_scores) + (KEYWORD_WEIGHT * kw_scores)

        # Get candidates (extra for re-ranking)
        n_candidates = top_k * 3 if use_reranker else top_k
        top_indices = np.argsort(combined)[::-1][:n_candidates]

        candidates = []
        for idx in top_indices:
            candidates.append({
                "chunk_id": int(idx),
                "text": self.chunks[idx]["text"],
                "source": self.chunks[idx]["source"],
                "semantic_score": float(semantic_scores[idx]),
                "keyword_score": float(kw_scores[idx])
            })

        # Step 5: Re-rank
        if use_reranker:
            candidates = self._rerank(query, candidates)

        return candidates[:top_k], queries
