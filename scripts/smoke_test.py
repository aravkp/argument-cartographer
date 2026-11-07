"""Smoke test for the ingestion layer.

Usage:
    .venv/bin/python scripts/smoke_test.py <reddit_thread_url>

Fetches the thread and prints: total comments, max reply depth, and the
first 10 normalized comments.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.detector import UnsupportedURLError, detect
from src.ingestion.reddit import RedditFetchError, fetch_thread
from src.models import Comment


def max_depth(comments: list[Comment]) -> int:
    """Longest parent chain reachable within the fetched set (root = depth 0)."""
    by_id = {c.id: c for c in comments}
    depths: dict[str, int] = {}

    def depth_of(comment: Comment) -> int:
        if comment.id in depths:
            return depths[comment.id]
        parent = by_id.get(comment.parent_id) if comment.parent_id else None
        depths[comment.id] = 0 if parent is None else depth_of(parent) + 1
        return depths[comment.id]

    return max((depth_of(c) for c in comments), default=0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch a Reddit thread and summarize it.")
    parser.add_argument("url", help="Reddit thread URL")
    args = parser.parse_args()

    try:
        ref = detect(args.url)
    except UnsupportedURLError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Platform: {ref.platform}   thread_id: {ref.thread_id}   "
          f"subreddit: {ref.metadata.get('subreddit', '?')}")
    print("Fetching…")
    try:
        comments = fetch_thread(ref)
    except RedditFetchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\nTotal comments fetched: {len(comments)}")
    print(f"Max reply depth:        {max_depth(comments)}")
    print("\nFirst 10 normalized comments:")
    for i, c in enumerate(comments[:10], 1):
        body = textwrap.shorten(c.text.replace("\n", " "), width=100, placeholder="…")
        parent = c.parent_id or "(top-level)"
        print(f"\n{i:2}. [{c.score:>5} pts] u/{c.author}  id={c.id}  parent={parent}")
        print(f"    {c.timestamp:%Y-%m-%d %H:%M} UTC | {body}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
