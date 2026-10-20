"""Instruction -> embedding (all-MiniLM-L6-v2, L2-normalized, 384-D).

Known instructions come from embeddings.json (no encoder needed, e.g. on the training PC); new
wording is encoded on demand with sentence-transformers (Jetson .venv).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

TABLE = json.loads((Path(__file__).with_name('embeddings.json')).read_text())
DIM = TABLE['dim']


@lru_cache(maxsize=None)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(TABLE['encoder'], device='cpu')


def embed(text: str) -> np.ndarray:
    if text in TABLE['embeddings']:
        return np.asarray(TABLE['embeddings'][text], np.float32)
    return _model().encode([text], normalize_embeddings=True)[0].astype(np.float32)
