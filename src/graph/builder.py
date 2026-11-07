"""Argument graph construction and commenter stance clustering.

The graph has claim clusters as nodes and supports/attacks/restates edges.
Stance groups come from community detection over a commenter-affinity graph:
commenters accrue positive weight for asserting the same or mutually
supporting/restating claims, negative weight when their claims attack each
other; communities are found on the positive residual.
"""
from __future__ import annotations

from itertools import combinations

import networkx as nx

from src.models import ClaimCluster, ClaimEdge, Comment, StanceGroup

_MIN_GROUP_SIZE = 2


def build_graph(
    clusters: list[ClaimCluster], edges: list[ClaimEdge], comments: list[Comment]
) -> nx.DiGraph:
    """Assemble the claim graph: one node per cluster, typed directed edges."""
    author_of = {c.id: c.author for c in comments}
    graph = nx.DiGraph()
    for cluster in clusters:
        graph.add_node(
            cluster.id,
            text=cluster.text,
            assertions=len(cluster.claim_ids),
            comment_ids=cluster.comment_ids,
            authors=sorted({author_of[cid] for cid in cluster.comment_ids if cid in author_of}),
        )
    for edge in edges:
        if edge.source in graph and edge.target in graph:
            graph.add_edge(edge.source, edge.target, relation=edge.relation)
    return graph


def cluster_stances(
    clusters: list[ClaimCluster],
    edges: list[ClaimEdge],
    comments: list[Comment],
) -> list[StanceGroup]:
    """Group commenters by stance; returns groups of ≥2, largest first."""
    author_of = {c.id: c.author for c in comments}
    authors_of_cluster = {
        cluster.id: {
            author_of[cid]
            for cid in cluster.comment_ids
            if author_of.get(cid, "[deleted]") != "[deleted]"
        }
        for cluster in clusters
    }

    affinity: nx.Graph = nx.Graph()
    def add_weight(a: str, b: str, w: float) -> None:
        if a != b:
            affinity.add_edge(a, b, weight=affinity.get_edge_data(a, b, {}).get("weight", 0) + w)

    # Same cluster → strong agreement (they assert the same claim).
    for authors in authors_of_cluster.values():
        for a, b in combinations(sorted(authors), 2):
            add_weight(a, b, 2.0)

    # Cross-cluster relations transfer between the clusters' assert-ers.
    relation_weight = {"supports": 1.0, "restates": 1.0, "attacks": -2.0}
    for edge in edges:
        w = relation_weight[edge.relation]
        for a in authors_of_cluster.get(edge.source, ()):
            for b in authors_of_cluster.get(edge.target, ()):
                add_weight(a, b, w)

    # Community detection needs positive weights; drop net-negative ties
    # (an attack severing two commenters simply leaves no edge between them).
    positive = nx.Graph(
        (a, b, d) for a, b, d in affinity.edges(data=True) if d["weight"] > 0
    )
    if not positive.number_of_edges():
        return []

    communities = nx.community.greedy_modularity_communities(positive, weight="weight")
    groups = []
    for members in communities:
        if len(members) < _MIN_GROUP_SIZE:
            continue
        groups.append(sorted(members))
    groups.sort(key=len, reverse=True)
    return [
        StanceGroup(id=f"stance:{n}", label=f"Stance group {n + 1}", authors=authors)
        for n, authors in enumerate(groups)
    ]
