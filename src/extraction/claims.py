"""Claim extraction via Gemini (gemini-2.5-flash, structured output).

Comments are sent in batches; the model returns validated JSON mapping each
comment id to its atomic claims. Throttling/retry for the Gemini free tier
lives in src/llm.py: a 200-comment thread is 5 requests, ~1–3 minutes.
"""
from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from src.llm import LLMError, generate_structured, make_client
from src.models import Claim, Comment

_BATCH_SIZE = 40
_MAX_COMMENT_CHARS = 1500  # truncate walls of text; claims live up front anyway
_SKIP_BODIES = {"", "[removed]", "[deleted]"}

_PROMPT = """You are an argument-mining assistant. For each numbered comment below \
(from an online discussion), extract the atomic claims it asserts.

A claim is a single, self-contained declarative statement the commenter puts \
forward as true or advocates for — factual assertions, evaluations, or \
recommendations. Rephrase into standalone form (resolve "this/it/that" using \
the comment's own context). Do NOT extract questions, greetings, jokes without \
an assertion, or pure expressions of emotion. 0 claims for such comments is \
correct and common. Extract at most 3 claims per comment, most central first.

Return one entry per comment, keyed by its comment_id, in the order given.

Comments:
{comments}"""

# Kept as the public alias so callers can surface extraction failures.
ClaimExtractionError = LLMError


class _CommentClaims(BaseModel):
    comment_id: str
    claims: list[str]


class _ExtractionBatch(BaseModel):
    results: list[_CommentClaims]


def extract_claims(
    comments: list[Comment],
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Claim]:
    """Extract atomic claims from comments; returns one Claim per assertion.

    on_progress, if given, is called with (batches_done, batches_total).
    """
    client = make_client()
    usable = [c for c in comments if c.text.strip() not in _SKIP_BODIES]
    batches = [usable[i : i + _BATCH_SIZE] for i in range(0, len(usable), _BATCH_SIZE)]

    claims: list[Claim] = []
    for n, batch in enumerate(batches):
        listing = "\n\n".join(
            f"[comment_id: {c.id}]\n{c.text[:_MAX_COMMENT_CHARS]}" for c in batch
        )
        result = generate_structured(
            client, _PROMPT.format(comments=listing), _ExtractionBatch
        )
        claims.extend(_to_claims(batch, result))
        if on_progress:
            on_progress(n + 1, len(batches))
    return claims


def _to_claims(batch: list[Comment], result: _ExtractionBatch) -> list[Claim]:
    valid_ids = {c.id for c in batch}
    claims = []
    for entry in result.results:
        if entry.comment_id not in valid_ids:
            continue  # model hallucinated an id — drop rather than mis-attribute
        for n, text in enumerate(entry.claims[:3]):
            if text.strip():
                claims.append(
                    Claim(
                        id=f"{entry.comment_id}:{n}",
                        comment_id=entry.comment_id,
                        text=text.strip(),
                    )
                )
    return claims
