"""SQLite cache for fetched threads, keyed by a hash of the thread URL."""
from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

from pydantic import TypeAdapter

from src.models import Claim, ClaimCluster, ClaimEdge, Comment, StanceGroup

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "cache.db"

_comments_adapter = TypeAdapter(list[Comment])
_claims_adapter = TypeAdapter(list[Claim])
_clusters_adapter = TypeAdapter(list[ClaimCluster])
_edges_adapter = TypeAdapter(list[ClaimEdge])
_stances_adapter = TypeAdapter(list[StanceGroup])


def _url_key(url: str) -> str:
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS threads (
            url_hash   TEXT PRIMARY KEY,
            url        TEXT NOT NULL,
            comments   TEXT NOT NULL,
            fetched_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            url_hash     TEXT PRIMARY KEY,
            url          TEXT NOT NULL,
            claims       TEXT NOT NULL,
            clusters     TEXT NOT NULL,
            analyzed_at  REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS graphs (
            url_hash  TEXT PRIMARY KEY,
            url       TEXT NOT NULL,
            edges     TEXT NOT NULL,
            stances   TEXT NOT NULL,
            built_at  REAL NOT NULL
        )
        """
    )
    return conn


def get_comments(
    url: str, max_age_s: float | None = None, db_path: Path = DEFAULT_DB_PATH
) -> list[Comment] | None:
    """Return cached comments for a thread URL, or None on a miss.

    max_age_s, if given, treats entries older than that many seconds as misses.
    """
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT comments, fetched_at FROM threads WHERE url_hash = ?",
            (_url_key(url),),
        ).fetchone()
    if row is None:
        return None
    payload, fetched_at = row
    if max_age_s is not None and time.time() - fetched_at > max_age_s:
        return None
    return _comments_adapter.validate_json(payload)


def set_comments(url: str, comments: list[Comment], db_path: Path = DEFAULT_DB_PATH) -> None:
    """Store (or replace) the fetched comments for a thread URL."""
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO threads (url_hash, url, comments, fetched_at) "
            "VALUES (?, ?, ?, ?)",
            (
                _url_key(url),
                url,
                _comments_adapter.dump_json(comments).decode("utf-8"),
                time.time(),
            ),
        )


def get_analysis(
    url: str, db_path: Path = DEFAULT_DB_PATH
) -> tuple[list[Claim], list[ClaimCluster]] | None:
    """Return cached (claims, clusters) for a thread URL, or None on a miss.

    Analyses have no max age: extraction is deterministic-ish and costs LLM
    calls, so a cached analysis stays valid until the thread is re-fetched.
    """
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT claims, clusters FROM analyses WHERE url_hash = ?",
            (_url_key(url),),
        ).fetchone()
    if row is None:
        return None
    return (
        _claims_adapter.validate_json(row[0]),
        _clusters_adapter.validate_json(row[1]),
    )


def get_graph(
    url: str, db_path: Path = DEFAULT_DB_PATH
) -> tuple[list[ClaimEdge], list[StanceGroup]] | None:
    """Return cached (edges, stance groups) for a thread URL, or None on a miss."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT edges, stances FROM graphs WHERE url_hash = ?",
            (_url_key(url),),
        ).fetchone()
    if row is None:
        return None
    return (
        _edges_adapter.validate_json(row[0]),
        _stances_adapter.validate_json(row[1]),
    )


def set_graph(
    url: str,
    edges: list[ClaimEdge],
    stances: list[StanceGroup],
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    """Store (or replace) the inferred edges + stance groups for a thread URL."""
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO graphs (url_hash, url, edges, stances, built_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                _url_key(url),
                url,
                _edges_adapter.dump_json(edges).decode("utf-8"),
                _stances_adapter.dump_json(stances).decode("utf-8"),
                time.time(),
            ),
        )


def set_analysis(
    url: str,
    claims: list[Claim],
    clusters: list[ClaimCluster],
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    """Store (or replace) the extraction + dedup results for a thread URL."""
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO analyses (url_hash, url, claims, clusters, analyzed_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                _url_key(url),
                url,
                _claims_adapter.dump_json(claims).decode("utf-8"),
                _clusters_adapter.dump_json(clusters).decode("utf-8"),
                time.time(),
            ),
        )
