"""Validate and parse discussion-thread URLs.

Each supported platform contributes one parser function that inspects a URL
and returns a ThreadRef if (and only if) the URL is a thread on that
platform. To add a platform later, write a parser and append it to PARSERS —
nothing else changes.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from urllib.parse import urlparse

from src.models import ThreadRef


class UnsupportedURLError(ValueError):
    """Raised when a URL is not a thread on any supported platform."""


_REDDIT_HOSTS = {
    "reddit.com",
    "www.reddit.com",
    "old.reddit.com",
    "new.reddit.com",
    "np.reddit.com",
    "m.reddit.com",
    "sh.reddit.com",
}

# /r/<subreddit>/comments/<thread_id>[/<slug>[/...]]
_REDDIT_COMMENTS_RE = re.compile(
    r"^/r/(?P<subreddit>\w+)/comments/(?P<thread_id>[a-z0-9]+)(?:/|$)", re.IGNORECASE
)
# Bare permalink without subreddit: /comments/<thread_id>
_REDDIT_BARE_RE = re.compile(r"^/comments/(?P<thread_id>[a-z0-9]+)(?:/|$)", re.IGNORECASE)
# Share links: /r/<subreddit>/s/<share_code> — need a redirect to resolve.
_REDDIT_SHARE_RE = re.compile(r"^/r/(?P<subreddit>\w+)/s/\w+/?$", re.IGNORECASE)


def _parse_reddit(url: str) -> ThreadRef | None:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path

    if host == "redd.it":
        thread_id = path.strip("/").split("/")[0]
        if thread_id:
            return ThreadRef(platform="reddit", url=url, thread_id=thread_id.lower())
        return None

    if host not in _REDDIT_HOSTS:
        return None

    if match := _REDDIT_COMMENTS_RE.match(path):
        return ThreadRef(
            platform="reddit",
            url=url,
            thread_id=match["thread_id"].lower(),
            metadata={"subreddit": match["subreddit"]},
        )
    if match := _REDDIT_BARE_RE.match(path):
        return ThreadRef(platform="reddit", url=url, thread_id=match["thread_id"].lower())
    if match := _REDDIT_SHARE_RE.match(path):
        # Share links hide the thread id behind a redirect; the fetcher
        # resolves it (thread_id=None signals that).
        return ThreadRef(
            platform="reddit",
            url=url,
            thread_id=None,
            metadata={"subreddit": match["subreddit"]},
        )
    return None


PARSERS: list[Callable[[str], ThreadRef | None]] = [
    _parse_reddit,
]

SUPPORTED_PLATFORMS = ["reddit"]


def detect(url: str) -> ThreadRef:
    """Return a ThreadRef for a supported thread URL, or raise UnsupportedURLError."""
    url = url.strip()
    if not url:
        raise UnsupportedURLError("No URL provided. Paste a Reddit thread URL.")
    if "://" not in url:
        url = "https://" + url

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise UnsupportedURLError(f"'{url}' is not a valid URL.")

    for parser in PARSERS:
        if ref := parser(url):
            return ref

    raise UnsupportedURLError(
        f"'{url}' is not a supported discussion thread. Supported platforms: "
        f"{', '.join(SUPPORTED_PLATFORMS)}. For Reddit, paste a thread link like "
        "https://www.reddit.com/r/AskReddit/comments/abc123/some_title/"
    )
