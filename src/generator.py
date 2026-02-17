"""
Answer Generator
Takes retrieved chunks + question, sends to Claude for answer

ANALOGY: This is the reader. You hand them the index cards
(retrieved chunks) and your question. They read the cards
and write an answer using ONLY what's on those cards.
"""
import requests
from src.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL


def generate_answer(query, context_chunks):
    """
    Send retrieved chunks + question to Claude

    The system prompt restricts Claude to ONLY use the
    provided context — no outside knowledge allowed.
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
            "model": ANTHROPIC_MODEL,
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
    if "content" not in data:
        return f"API Error: {data.get('error', {}).get('message', 'Unknown error')}"

    return data["content"][0]["text"]
