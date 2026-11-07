"""Pydantic models shared across module boundaries."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class Comment(BaseModel):
    """A single normalized comment from any platform.

    parent_id is None for top-level comments (direct replies to the post),
    otherwise it is the id of the parent Comment. Ids are bare (no platform
    prefixes like Reddit's "t1_").
    """

    id: str
    author: str
    text: str
    parent_id: str | None = None
    timestamp: datetime
    score: int = 0
    platform: str = "reddit"


class Claim(BaseModel):
    """An atomic claim extracted from a comment."""

    id: str
    comment_id: str
    text: str


class ClaimCluster(BaseModel):
    """A group of semantically equivalent claims, deduplicated.

    text is the canonical phrasing (the cluster's medoid claim);
    claim_ids/comment_ids record every claim and commenter that asserted it.
    """

    id: str
    text: str
    claim_ids: list[str]
    comment_ids: list[str]


Relation = Literal["supports", "attacks", "restates"]


class ClaimEdge(BaseModel):
    """A typed, directed relation between two claim clusters: source → target."""

    source: str
    target: str
    relation: Relation


class StanceGroup(BaseModel):
    """A group of commenters holding a similar position in the thread."""

    id: str
    label: str
    authors: list[str]


class ThreadRef(BaseModel):
    """A validated reference to a discussion thread on a supported platform.

    thread_id may be None for URL forms that require a network round-trip to
    resolve (e.g. Reddit share links); the platform fetcher resolves those.
    """

    platform: str
    url: str
    thread_id: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


def utc_from_epoch(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc)
