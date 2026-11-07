"""Semantic deduplication of claims.

Claims are embedded locally with sentence-transformers (all-MiniLM-L6-v2)
and grouped by agglomerative clustering on cosine distance; each cluster's
canonical phrasing is its medoid (the claim closest to every other member).

Despite the filename, no Pinecone is involved: at v1 scale (a few hundred
claims per thread) local embeddings + scikit-learn clustering are exact,
free, and fast. If claim volumes ever outgrow memory, swap the internals
for a Pinecone index behind the same dedupe_claims() signature.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from src.models import Claim, ClaimCluster

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Cosine distance under which two claims count as the same claim. 0.0 =
# identical embeddings; ~0.3 tolerates paraphrase while keeping opposing
# claims (which often share topic words) apart.
DISTANCE_THRESHOLD = 0.3

_model = None


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed texts with the project's sentence-transformer (L2-normalized)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # slow import — defer

        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model.encode(texts, normalize_embeddings=True)


def dedupe_claims(claims: list[Claim]) -> list[ClaimCluster]:
    """Group semantically equivalent claims; returns one ClaimCluster per group.

    Singleton clusters are included, so every input claim appears in exactly
    one cluster. Clusters are ordered by size (most-asserted claims first).
    """
    if not claims:
        return []
    if len(claims) == 1:
        return [_make_cluster(0, claims, np.zeros((1, 1)))]

    embeddings = embed_texts([c.text for c in claims])
    labels = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=DISTANCE_THRESHOLD,
    ).fit_predict(embeddings)

    clusters = []
    for label in sorted(set(labels)):
        member_idx = np.flatnonzero(labels == label)
        members = [claims[i] for i in member_idx]
        clusters.append(_make_cluster(label, members, embeddings[member_idx]))

    clusters.sort(key=lambda c: len(c.claim_ids), reverse=True)
    return [c.model_copy(update={"id": f"cluster:{n}"}) for n, c in enumerate(clusters)]


def _make_cluster(label: int, members: list[Claim], embeddings: np.ndarray) -> ClaimCluster:
    # Medoid: member with the highest mean similarity to the rest.
    medoid = members[int(np.argmax((embeddings @ embeddings.T).mean(axis=1)))]
    return ClaimCluster(
        id=f"cluster:{label}",
        text=medoid.text,
        claim_ids=[c.id for c in members],
        comment_ids=sorted({c.comment_id for c in members}),
    )
