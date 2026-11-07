"""Extraction-layer test: claims + dedup on a real thread.

Usage:
    .venv/bin/python scripts/extraction_test.py <reddit_thread_url> [--max-comments N]

Fetches the thread (via cache when possible), extracts claims with Gemini,
dedupes them, and prints stats plus the largest claim clusters.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import cache
from src.dedup.pinecone_dedup import dedupe_claims
from src.extraction.claims import ClaimExtractionError, extract_claims
from src.ingestion.detector import UnsupportedURLError, detect
from src.ingestion.reddit import RedditFetchError, fetch_thread


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Reddit thread URL")
    parser.add_argument("--max-comments", type=int, default=None,
                        help="only analyze the first N comments (default: all)")
    args = parser.parse_args()

    try:
        ref = detect(args.url)
    except UnsupportedURLError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    comments = cache.get_comments(args.url)
    if comments is None:
        print("Fetching thread…")
        try:
            comments = fetch_thread(ref)
        except RedditFetchError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        cache.set_comments(args.url, comments)
    else:
        print("Comments loaded from cache.")
    if args.max_comments:
        comments = comments[: args.max_comments]

    print(f"Extracting claims from {len(comments)} comments with Gemini…")
    start = time.time()
    try:
        claims = extract_claims(
            comments,
            on_progress=lambda done, total: print(f"  batch {done}/{total}"),
        )
    except ClaimExtractionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Extraction took {time.time() - start:.1f}s")

    print("Deduplicating…")
    clusters = dedupe_claims(claims)
    if args.max_comments is None:
        cache.set_analysis(args.url, claims, clusters)

    multi = [c for c in clusters if len(c.claim_ids) > 1]
    print(f"\nComments analyzed:        {len(comments)}")
    print(f"Claims extracted:         {len(claims)}")
    print(f"Distinct claims (deduped): {len(clusters)}")
    print(f"Clusters with >1 member:  {len(multi)}")

    print("\nLargest claim clusters:")
    for cluster in clusters[:8]:
        n = len(cluster.claim_ids)
        if n < 2:
            break
        print(f"\n  ({n}× from {len(cluster.comment_ids)} comments) "
              + textwrap.shorten(cluster.text, 110, placeholder="…"))

    print("\nSample extracted claims:")
    for claim in claims[:8]:
        print(f"  [{claim.comment_id}] " + textwrap.shorten(claim.text, 110, placeholder="…"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
