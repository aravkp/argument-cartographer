"""Graph-layer test: edge inference + stance clustering on a real thread.

Usage:
    .venv/bin/python scripts/graph_test.py <reddit_thread_url>

Requires the thread to have a cached analysis (run extraction_test.py first,
or the Streamlit app). Prints edges by type and stance groups, and writes an
interactive HTML map to data/graph_<thread_id>.html.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import cache
from src.graph.builder import build_graph, cluster_stances
from src.graph.edges import infer_edges
from src.graph.render import graph_to_html
from src.ingestion.detector import detect
from src.llm import LLMError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Reddit thread URL (must be analyzed already)")
    args = parser.parse_args()

    ref = detect(args.url)
    comments = cache.get_comments(args.url)
    analysis = cache.get_analysis(args.url)
    if comments is None or analysis is None:
        print(
            "ERROR: no cached analysis for this URL — run "
            "scripts/extraction_test.py on it first.",
            file=sys.stderr,
        )
        return 1
    claims, clusters = analysis

    graph_data = cache.get_graph(args.url)
    if graph_data is None:
        print(f"Inferring relations between {len(clusters)} distinct claims…")
        try:
            edges = infer_edges(
                clusters,
                comments,
                on_progress=lambda done, total: print(f"  batch {done}/{total}"),
            )
        except LLMError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        stances = cluster_stances(clusters, edges, comments)
        cache.set_graph(args.url, edges, stances)
    else:
        print("Graph loaded from cache.")
        edges, stances = graph_data

    graph = build_graph(clusters, edges, comments)
    by_id = {c.id: c for c in clusters}

    print(f"\nEdges found: {len(edges)}")
    for relation in ("attacks", "supports", "restates"):
        subset = [e for e in edges if e.relation == relation]
        print(f"\n--- {relation} ({len(subset)}) ---")
        for edge in subset[:6]:
            src = textwrap.shorten(by_id[edge.source].text, 70, placeholder="…")
            dst = textwrap.shorten(by_id[edge.target].text, 70, placeholder="…")
            print(f"  • {src}\n      → {dst}")

    print(f"\nStance groups: {len(stances)}")
    for group in stances:
        print(f"  {group.label}: {', '.join(group.authors[:8])}"
              + (" …" if len(group.authors) > 8 else ""))

    out = Path(__file__).resolve().parent.parent / "data" / f"graph_{ref.thread_id}.html"
    out.write_text(graph_to_html(graph, stances))
    print(f"\nInteractive map written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
