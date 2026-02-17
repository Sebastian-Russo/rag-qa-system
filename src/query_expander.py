"""
Query Expansion
Takes a user question and generates alternative phrasings
to improve retrieval coverage

ANALOGY: A translator who rephrases your question multiple
ways before the librarian searches. Catches cases where your
wording doesn't match how the books phrase things.
"""
import requests
from src.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL


def expand_query(query):
    """
    Generate alternative phrasings using Claude

    Input:  "what spell made a deer for harry"
    Output: [
        "what spell made a deer for harry",
        "Expecto Patronum conjured a silver deer",
        "Harry's patronus took the shape of a stag",
        "A silvery stag emerged from Harry's wand"
    ]
    """
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        },
        json={
            "model": ANTHROPIC_MODEL,
            "max_tokens": 256,
            "system": """Generate 3 alternative phrasings of the user's question
for searching through Harry Potter book text. Think about how the
information would actually be written in the books. Return ONLY the
3 phrasings, one per line, no numbering or extra text.""",
            "messages": [{"role": "user", "content": query}]
        }
    )

    data = response.json()
    if "content" not in data:
        return [query]

    expanded = data["content"][0]["text"].strip().split("\n")
    expanded = [q.strip() for q in expanded if q.strip()]

    # Always include original query
    return [query] + expanded
