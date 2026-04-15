"""
RAG-based knowledge service for the Voice Sales Agent.

When a PDF is uploaded:
  1. Text is split into overlapping chunks (~400 tokens each).
  2. Every chunk is embedded with text-embedding-3-small.
  3. Embeddings are stored in memory as a numpy matrix.

At query time:
  • The user's transcript is embedded.
  • Cosine similarity is computed against all chunk embeddings.
  • The top-K most relevant chunks are returned as formatted context.
  • A keyword hint is built for Whisper's STT prompt (domain vocabulary).
"""

import os
import re
import logging
from typing import Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chunker — paragraph-aware, with overlap
# ---------------------------------------------------------------------------

_CHUNK_CHARS    = 1_500   # ~375 tokens per chunk — more context per chunk
_CHUNK_OVERLAP  = 200     # chars of overlap between consecutive chunks


def _split_chunks(text: str) -> list[str]:
    """
    Split on paragraph / section boundaries first, then fall back to
    character-based sliding window so no chunk exceeds _CHUNK_CHARS.
    """
    # Normalise whitespace
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Split on double-newlines (paragraph breaks)
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current = ""

    for para in paras:
        # If adding this paragraph keeps us under the limit, accumulate
        if len(current) + len(para) + 2 <= _CHUNK_CHARS:
            current = (current + "\n\n" + para).strip()
        else:
            # Flush the current chunk
            if current:
                chunks.append(current)
            # If a single paragraph is longer than the limit, slice it
            if len(para) > _CHUNK_CHARS:
                start = 0
                while start < len(para):
                    chunks.append(para[start : start + _CHUNK_CHARS])
                    start += _CHUNK_CHARS - _CHUNK_OVERLAP
                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    return chunks


# ---------------------------------------------------------------------------
# KnowledgeBase
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """
    In-memory RAG store.  No external dependencies beyond openai + numpy.
    """

    EMBED_MODEL  = "text-embedding-3-small"
    TOP_K        = 7       # chunks to retrieve per query
    MIN_SIM      = 0.25    # similarity threshold — lower catches borderline-relevant chunks

    def __init__(self):
        self._chunks: list[str] = []
        self._embeddings         = None   # numpy ndarray shape (N, D)
        self._keywords: list[str] = []    # top nouns/phrases for STT hint
        self._client             = None   # lazy-init to avoid import cycle

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, text: str) -> int:
        """
        Chunk, embed and index a document.
        Returns the number of chunks created.
        Blocks (sync) — call from a thread executor.
        """
        import numpy as np

        chunks = _split_chunks(text)
        if not chunks:
            log.warning("KnowledgeBase.ingest: no chunks extracted")
            return 0

        log.info("KnowledgeBase: embedding %d chunks …", len(chunks))
        client = self._get_client()

        # Batch in groups of 100 (API limit)
        all_vecs = []
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            resp  = client.embeddings.create(model=self.EMBED_MODEL, input=batch)
            all_vecs.extend([e.embedding for e in resp.data])

        self._chunks     = chunks
        self._embeddings = np.array(all_vecs, dtype=np.float32)

        # Build keyword hint for STT (capitalised tokens, numbers, special words)
        self._keywords = _extract_keywords(text)

        log.info(
            "KnowledgeBase: indexed %d chunks, %d keywords",
            len(chunks), len(self._keywords),
        )
        return len(chunks)

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_overlap(query: str, chunk: str) -> float:
        """
        Fraction of query tokens (≥3 chars) that appear in the chunk.
        Returns a score in [0, 1] to blend with cosine similarity.
        """
        q_tokens = {t.lower() for t in re.findall(r"\b\w{3,}\b", query)}
        if not q_tokens:
            return 0.0
        c_lower = chunk.lower()
        hits = sum(1 for t in q_tokens if t in c_lower)
        return hits / len(q_tokens)

    def retrieve(self, query: str, top_k: int | None = None) -> list[str]:
        """
        Hybrid retrieval: cosine similarity (semantic) + keyword overlap (lexical).
        Returns up to top_k most relevant chunks above MIN_SIM threshold.
        """
        if self._embeddings is None or not self._chunks:
            return []

        import numpy as np

        k = top_k or self.TOP_K
        client = self._get_client()

        q_resp = client.embeddings.create(model=self.EMBED_MODEL, input=[query])
        q_vec  = np.array(q_resp.data[0].embedding, dtype=np.float32)

        # Cosine similarity
        norms  = np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(q_vec)
        norms  = np.where(norms == 0, 1e-9, norms)
        cosine = (self._embeddings @ q_vec) / norms

        # Hybrid score: 90 % semantic + 10 % keyword overlap
        kw_scores = np.array(
            [self._keyword_overlap(query, c) for c in self._chunks],
            dtype=np.float32,
        )
        hybrid = 0.90 * cosine + 0.10 * kw_scores

        top_idx = np.argsort(hybrid)[::-1][:k]
        return [
            self._chunks[i]
            for i in top_idx
            if cosine[i] >= self.MIN_SIM   # gate on semantic sim, not hybrid
        ]

    def get_context(self, query: str) -> str:
        """
        Return a formatted string of retrieved chunks to inject into the LLM.
        Returns empty string if knowledge base is empty or nothing matches.
        """
        chunks = self.retrieve(query)
        if not chunks:
            return ""
        return "\n\n---\n\n".join(chunks)

    # ------------------------------------------------------------------
    # STT domain hint
    # ------------------------------------------------------------------

    def get_stt_hint(self) -> str:
        """
        Short comma-separated string of key product/brand terms
        to use as Whisper's prompt= parameter for better transcription.
        Returns "" if no knowledge is loaded.
        """
        if not self._keywords:
            return ""
        return ", ".join(self._keywords[:40])   # Whisper prompt cap ~250 tokens

    @property
    def is_loaded(self) -> bool:
        return self._embeddings is not None and len(self._chunks) > 0


# ---------------------------------------------------------------------------
# Keyword extractor — for STT domain hint
# ---------------------------------------------------------------------------

def _extract_keywords(text: str) -> list[str]:
    """
    Extract likely product names, brands, prices and numbers from the text.
    Heuristic: capitalised words, currency patterns, model numbers.
    """
    # Capitalised multi-word phrases (brand / product names)
    cap_phrases = re.findall(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*\b", text)
    # Numbers + units (₹ 50,000 / 4K / 5G / 128GB)
    units = re.findall(r"(?:₹|Rs\.?)\s?\d[\d,]*|\b\d+\s?(?:GB|TB|MP|Hz|W|K|G|inch)\b", text, re.I)
    # Model numbers  (SM-A155, iPhone 15, Galaxy S24)
    models = re.findall(r"\b[A-Z]{1,5}[\-\s]?\d[\w\-]*\b", text)

    seen = set()
    result = []
    for kw in cap_phrases + units + models:
        kw = kw.strip()
        if kw and kw.lower() not in seen and len(kw) > 2:
            seen.add(kw.lower())
            result.append(kw)

    return result[:80]   # cap list length


# ---------------------------------------------------------------------------
# Note: KnowledgeBase instances are now created per-tenant in TenantConfig.
# The global singleton is intentionally removed; use tenant.knowledge_base.
# ---------------------------------------------------------------------------
