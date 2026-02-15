# Types of RAG Systems

**Basic RAG** — what most people build first. You chunk documents, embed them, store in a vector database, retrieve relevant chunks on query, pass to LLM for answer. Simple pipeline: question → search → answer. This is what we'll build.

**Conversational RAG** — same as basic but with chat history. It remembers previous questions so you can ask follow-ups like "what about the second book?" without repeating context. Adds a memory layer.

**Multi-modal RAG** — handles not just text but images, tables, PDFs with charts. Useful for technical documents where a diagram is the answer, not a paragraph.

**Agentic RAG** — the RAG system decides how to search. Instead of one simple retrieval, it might rephrase your question, search multiple times, filter results, or decide it needs a different source entirely. This overlaps with our agent project later.

**Hybrid RAG** — combines keyword search (traditional, like Ctrl+F) with semantic search (vector similarity, meaning-based). Sometimes the exact word match matters more than meaning.

For our project, we'll build basic RAG with room to extend to conversational if you want.

## Common Types of Data People Load In

### Personal/private use:

Books and textbooks you own (your Harry Potter idea)
Class notes, lecture PDFs
Personal journals or writing
Tax documents, medical records, legal contracts

### Professional:

Company documentation, SOPs, internal wikis
Codebase documentation
Slack/email archives
Meeting transcripts

### Technical:

API docs, technical manuals
Research papers
Programming language docs

### Creative:

World-building docs for D&D campaigns or fiction writing
Recipe collections
Travel journals/guides

The whole point of RAG is it works with whatever you throw at it — PDFs, text files, markdown, even web pages. The system doesn't care about the content, it just chunks it, embeds it, and retrieves what's relevant.

What sounds useful to you? You mentioned Harry Potter before — we could also do something you'd actually keep using after the project, like a personal knowledge base with documents you already have.