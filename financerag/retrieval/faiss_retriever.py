"""FAISS-backed retrieval wrapper for the FinanceRAG project.

This module provides a `FaissRetriever` class that implements the `Retrieval`
protocol used across the repo. It supports two main flows:

- Build an index from an `Encoder` (e.g. `SentenceTransformerEncoder`) by
  encoding the corpus once and storing embeddings + ids + FAISS index on disk.
- Load a prebuilt FAISS index + ids and perform fast ANN search at query time.

Notes:
- Requires `faiss` (cpu or gpu). Install with `pip install faiss-cpu`.
- Embeddings are expected as float32 numpy arrays. For cosine similarity we
  normalize embeddings and use an inner-product index (`IndexFlatIP`).

Example (build index):

    from financerag.retrieval.sent_encoder import SentenceTransformerEncoder
    from financerag.retrieval.faiss_retriever import FaissRetriever

    encoder = SentenceTransformerEncoder("all-MiniLM-L6-v2")
    retriever = FaissRetriever(encoder=encoder)
    retriever.build_index(corpus_list, corpus_ids, index_path="corpus.index", emb_path="corpus_emb.npy", normalize=True)

Example (query):

    retriever = FaissRetriever(index_path="corpus.index", ids_path="corpus_ids.json")
    results = retriever.retrieve(corpus_dict, queries_dict, top_k=10)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import faiss
except Exception as e:  # pragma: no cover - faiss optional
    faiss = None

from financerag.common.protocols import Retrieval, Encoder

logger = logging.getLogger(__name__)


class FaissRetriever(Retrieval):
    """
    FAISS-backed retriever implementing the `Retrieval` protocol.

    Parameters
    - encoder: optional Encoder instance used to encode corpus if no index exists.
    - index_path / ids_path: optional paths to load a prebuilt faiss index and ids list.
    - normalize: whether to L2-normalize embeddings (useful for cosine similarity).
    """

    def __init__(self, encoder: Optional[Encoder] = None, index_path: Optional[str] = None, ids_path: Optional[str] = None, normalize: bool = True):
        if faiss is None:
            raise ImportError("faiss is not installed. Install faiss-cpu or faiss-gpu to use FaissRetriever.")

        self.encoder = encoder
        self.index_path = index_path
        self.ids_path = ids_path
        self.normalize = normalize

        self.index: Optional[faiss.Index] = None
        self.corpus_ids: Optional[List[str]] = None

        if index_path and ids_path:
            self._load_index(index_path, ids_path)

    # -- index build/load utilities --
    def build_index(self, corpus: List[Dict[str, str]], corpus_ids: List[str], index_path: Optional[str] = None, emb_path: Optional[str] = None, batch_size: int = 64, use_gpu: bool = False, normalize: Optional[bool] = None):
        """
        Build a FAISS index from a corpus and optional encoder.

        Args:
            corpus: list of documents in the same form used by SentenceTransformerEncoder (dicts with 'title' and 'text').
            corpus_ids: list of document IDs with the same order as corpus.
            index_path: optional path to save the built FAISS index.
            emb_path: optional path to save the computed embeddings (.npy).
            batch_size: encoder batch size.
            use_gpu: whether to try to use faiss GPU (if installed and available).
            normalize: if True, L2-normalize embeddings to use cosine similarity with IndexFlatIP.
        """
        if self.encoder is None:
            raise ValueError("An encoder is required to build index unless you provide precomputed embeddings.")

        logger.info("Computing corpus embeddings with encoder...")
        embeddings = self.encoder.encode_corpus(corpus, batch_size=batch_size)
        embeddings = np.asarray(embeddings, dtype=np.float32)

        if normalize if normalize is not None else self.normalize:
            faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        logger.info(f"Building FAISS IndexFlatIP (dim={dim})...")
        index = faiss.IndexFlatIP(dim)

        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception:
                logger.warning("Could not use FAISS GPU; falling back to CPU index.")

        index.add(embeddings)

        self.index = index
        self.corpus_ids = corpus_ids

        if emb_path:
            np.save(emb_path, embeddings)
        if index_path:
            faiss.write_index(index if not use_gpu else faiss.index_gpu_to_cpu(index), index_path)
        if self.ids_path:
            with open(self.ids_path, "w") as f:
                json.dump(self.corpus_ids, f)

        logger.info("FAISS index built and saved.")

    def _load_index(self, index_path: str, ids_path: str):
        p_index = Path(index_path)
        p_ids = Path(ids_path)
        if not p_index.exists() or not p_ids.exists():
            raise ValueError(f"Index file {index_path} or ids file {ids_path} not found.")

        logger.info(f"Loading FAISS index from {index_path} and ids from {ids_path}...")
        self.index = faiss.read_index(index_path)
        with open(ids_path, "r") as f:
            self.corpus_ids = json.load(f)

    # -- Retrieval protocol implementation --
    def retrieve(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], top_k: Optional[int] = 10, **kwargs) -> Dict[str, Dict[str, float]]:
        """
        Retrieve top-k documents for each query using FAISS ANN search.

        Notes: this method does not re-encode the corpus. It expects an index to be built or loaded.
        If no index is present but an encoder was provided, it will build an in-memory index (not persisted) before querying.
        """
        if self.index is None:
            # try to build an in-memory index from provided encoder and corpus list
            if self.encoder is None:
                raise ValueError("No FAISS index loaded and no encoder available to build one.")

            # Build corpus_list and ids from `corpus` dict ordering
            sorted_ids = list(corpus.keys())
            corpus_list = [corpus[cid] for cid in sorted_ids]
            self.build_index(corpus_list, sorted_ids, index_path=None, emb_path=None, batch_size=kwargs.get("batch_size", 64), normalize=self.normalize)

        # Prepare queries
        q_ids = list(queries.keys())
        q_texts = [queries[qid] for qid in q_ids]

        # Encode queries
        if self.encoder is None:
            raise ValueError("Encoder required to encode queries when using FaissRetriever.")
        q_emb = self.encoder.encode_queries(q_texts, batch_size=kwargs.get("batch_size", 16))
        q_emb = np.asarray(q_emb, dtype=np.float32)
        if self.normalize:
            faiss.normalize_L2(q_emb)

        # Search
        D, I = self.index.search(q_emb, top_k)

        results: Dict[str, Dict[str, float]] = {qid: {} for qid in q_ids}
        for i, qid in enumerate(q_ids):
            for score, idx in zip(D[i].tolist(), I[i].tolist()):
                if idx < 0 or idx >= len(self.corpus_ids):
                    continue
                doc_id = self.corpus_ids[idx]
                results[qid][doc_id] = float(score)

        return results
