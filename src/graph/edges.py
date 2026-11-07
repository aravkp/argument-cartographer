"""Infer supports/attacks/restates edges between claim clusters via Gemini.

Classifying every pair of clusters is quadratic (267 clusters → ~35k pairs),
so only candidate pairs are sent to the model:

1. reply pairs — cluster B holds a claim from a comment that replied to a
   comment whose claim is in cluster A (people arguing with each other), and
2. semantic pairs — cluster embeddings with cosine similarity in a band just
   below the dedup-merge threshold (same topic, not the same claim).

Candidates are capped and batched; the model labels each pair
supports / attacks / restates / none, and "none" pairs are dropped.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import get_args

import numpy as np
from pydantic import BaseModel

from src.dedup.pinecone_dedup import embed_texts
from src.llm import generate_structured, make_client
from src.models import ClaimCluster, ClaimEdge, Comment, Relation

_SIMILARITY_BAND = (0.45, 0.999)  # related topic, but below "same claim"
_MAX_PAIRS = 180
_PAIRS_PER_BATCH = 30

_PROMPT = """You are an argument-mining assistant. Below are pairs of claims taken \
from one online discussion. For each pair, classify the relationship of claim A \
toward claim B:

- "supports": A gives a reason, evidence, or agreement that strengthens B.
- "attacks": A contradicts, undermines, or argues against B.
- "restates": A expresses essentially the same position as B.
- "none": no meaningful argumentative relationship.

Judge only what the claims say — do not invent context. When in doubt, answer \
"none". Return one entry per pair, keyed by its pair_id, in the order given.

Pairs:
{pairs}"""


class _PairVerdict(BaseModel):
    pair_id: int
    relation: str


class _EdgeBatch(BaseModel):
    results: list[_PairVerdict]


def infer_edges(
    clusters: list[ClaimCluster],
    comments: list[Comment],
    on_progress: Callable[[int, int], None] | None = None,
) -> list[ClaimEdge]:
    """Classify candidate cluster pairs; returns typed, directed edges."""
    pairs = _candidate_pairs(clusters, comments)
    if not pairs:
        return []

    client = make_client()
    batches = [pairs[i : i + _PAIRS_PER_BATCH] for i in range(0, len(pairs), _PAIRS_PER_BATCH)]
    by_id = {c.id: c for c in clusters}

    edges: list[ClaimEdge] = []
    for n, batch in enumerate(batches):
        listing = "\n\n".join(
            f"[pair_id: {i}]\nA: {by_id[a].text}\nB: {by_id[b].text}"
            for i, (a, b) in enumerate(batch)
        )
        result = generate_structured(client, _PROMPT.format(pairs=listing), _EdgeBatch)
        for verdict in result.results:
            if 0 <= verdict.pair_id < len(batch) and verdict.relation in get_args(Relation):
                source, target = batch[verdict.pair_id]
                edges.append(
                    ClaimEdge(source=source, target=target, relation=verdict.relation)
                )
        if on_progress:
            on_progress(n + 1, len(batches))
    return _drop_duplicate_edges(edges)


def _candidate_pairs(
    clusters: list[ClaimCluster], comments: list[Comment]
) -> list[tuple[str, str]]:
    """Ordered (source, target) cluster-id pairs worth asking the model about."""
    pairs: dict[tuple[str, str], None] = {}  # insertion-ordered set

    # Reply pairs: source = the replying comment's cluster, target = parent's.
    parent_of = {c.id: c.parent_id for c in comments}
    clusters_of: dict[str, list[str]] = {}
    for cluster in clusters:
        for comment_id in cluster.comment_ids:
            clusters_of.setdefault(comment_id, []).append(cluster.id)
    for comment_id, cluster_ids in clusters_of.items():
        parent = parent_of.get(comment_id)
        if not parent:
            continue
        for source in cluster_ids:
            for target in clusters_of.get(parent, []):
                if source != target:
                    pairs.setdefault((source, target))

    # Semantic pairs: same topic (above band floor) but not dedup-identical.
    embeddings = embed_texts([c.text for c in clusters])
    similarity = embeddings @ embeddings.T
    lo, hi = _SIMILARITY_BAND
    upper = np.triu(similarity, k=1)
    for i, j in zip(*np.nonzero((upper >= lo) & (upper <= hi))):
        pair = (clusters[i].id, clusters[j].id)
        if pair not in pairs and pair[::-1] not in pairs:
            pairs.setdefault(pair)

    return list(pairs)[:_MAX_PAIRS]


def _drop_duplicate_edges(edges: list[ClaimEdge]) -> list[ClaimEdge]:
    seen: set[tuple[str, str]] = set()
    unique = []
    for edge in edges:
        if (edge.source, edge.target) not in seen:
            seen.add((edge.source, edge.target))
            unique.append(edge)
    return unique
