"""Shared Gemini access: client creation and throttled structured calls.

All LLM calls in the project go through generate_structured(), which paces
requests for the Gemini free tier (5 requests/minute on gemini-2.5-flash)
and retries rate-limit/transient errors with increasing waits. On a paid
tier this is merely a little slower than necessary.
"""
from __future__ import annotations

import os
import threading
import time
from typing import TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

GEMINI_MODEL = "gemini-2.5-flash"
_REQUEST_SPACING_S = 13  # ≈5/min, the free-tier RPM limit
_RETRY_WAITS_S = (20, 40, 60)

_last_request_at = 0.0
_throttle_lock = threading.Lock()

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when a Gemini call fails, with a human-readable reason."""


def make_client() -> genai.Client:
    load_dotenv()
    if not os.environ.get("GEMINI_API_KEY"):
        raise LLMError(
            "GEMINI_API_KEY is not set. Get a key at https://aistudio.google.com/apikey "
            "and put it in the project's .env file as: GEMINI_API_KEY=your-key-here"
        )
    return genai.Client()


def generate_structured(
    client: genai.Client, prompt: str, schema: type[SchemaT]
) -> SchemaT:
    """One structured-output Gemini call, throttled and retried."""
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        temperature=0.0,
    )

    last_error: Exception | None = None
    for attempt in range(1 + len(_RETRY_WAITS_S)):
        if attempt:
            time.sleep(_RETRY_WAITS_S[attempt - 1])
        _wait_for_slot()
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt, config=config
            )
            if isinstance(response.parsed, schema):
                return response.parsed
            last_error = ValueError("Gemini returned no parseable structured output.")
        except errors.APIError as exc:
            if exc.code in (400, 401, 403):
                raise LLMError(
                    f"Gemini rejected the request ({exc.code}): {exc.message}. "
                    "Check that GEMINI_API_KEY in .env is valid."
                ) from exc
            last_error = exc  # 429/5xx → wait and retry
    raise LLMError(
        f"Gemini call failed after {len(_RETRY_WAITS_S)} retries: {last_error}. "
        "If this is a rate limit (429), the free tier also has a daily request cap — "
        "wait a bit or check https://ai.google.dev/gemini-api/docs/rate-limits"
    ) from last_error


def _wait_for_slot() -> None:
    """Space requests _REQUEST_SPACING_S apart, across threads."""
    global _last_request_at
    with _throttle_lock:
        wait = _last_request_at + _REQUEST_SPACING_S - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.time()
