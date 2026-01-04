"""
RAG Chain: Retrieval-Augmented Generation
"""

from typing import List, Dict
from langchain.schema import Document
from openai import OpenAI
import httpx
import os


# =========================
# System Prompt
# =========================
SYSTEM_PROMPT = """
You are a Retrieval-Augmented Generation (RAG) assistant.

Rules:
- Answer ONLY using the provided context
- If the user asks in Russian, answer in Russian
- If the user asks in English, answer in English
- If the context is in English and the question is in Russian, translate the answer
- If the answer is not in the context, say that you don't know
- Do NOT hallucinate
- Cite sources using [number]
"""


class RAGChain:
    def __init__(
        self,
        vectorstore,
        model: str = "gpt-4o-mini",
        top_k: int = 3,
    ):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found")

        self.vectorstore = vectorstore
        self.top_k = top_k

        # OpenAI client (without proxy)
        self.client = OpenAI(
            http_client=httpx.Client(trust_env=False)
        )

        self.model = os.getenv("MODEL_NAME", model)
        self.temperature = float(os.getenv("TEMPERATURE", 0))

    # =========================
    # Build context from docs
    # =========================
    def _build_context(self, docs: List[Document]) -> str:
        context_parts = []

        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            context_parts.append(
                f"[{i}] Source: {source}\n{doc.page_content}"
            )

        return "\n\n".join(context_parts)

    # =========================
    # Main RAG method
    # =========================
    def ask(self, question: str) -> Dict:
        # 1️⃣ Retrieve relevant documents
        docs = self.vectorstore.similarity_search(
            question, k=self.top_k
        )

        if not docs:
            return {
                "answer": "I don't know.",
                "sources": []
            }

        # 2️⃣ Build context
        context = self._build_context(docs)

        # 3️⃣ User prompt (без правил!)
        prompt = f"""
Context:
{context}

Question:
{question}

Answer:
"""

        # 4 Generate answer
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )

        answer = response.choices[0].message.content.strip()

        # 5 Collect unique sources (без дублей)
        sources = []
        seen_sources = set()
        source_id = 1

        for doc in docs:
            source = doc.metadata.get("source", "unknown")

            if source in seen_sources:
                continue

            seen_sources.add(source)

            sources.append({
                "id": source_id,
                "source": source,
                "preview": doc.page_content[:200]
            })

            source_id += 1

        # 6 Final response
        return {
            "answer": answer,
            "sources": sources
        }
