"""Fetch a Reddit thread's comment tree by parsing old.reddit.com HTML.

Why HTML and not the API: Reddit deprecated unauthenticated .json endpoints
in May 2026 and gates API app creation behind manual approval (Responsible
Builder Policy). Old Reddit's server-rendered comment pages remain publicly
served and carry the full reply tree — ids, authors, scores, timestamps, and
nesting. One thread fetch is a single HTTP request (limit=500 puts up to 500
comments on one page, comfortably above our 200-comment cap).

If API access is approved later, a PRAW-based fetcher can replace this module
as long as it keeps the fetch_thread() signature. Be a good citizen: this is
a personal, low-volume tool — one request per thread, browser User-Agent,
results cached in SQLite (src/cache.py).
"""
from __future__ import annotations

from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup, Tag

from src.models import Comment, ThreadRef

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
COMMENT_CAP = 200  # v1 guardrail: keep the top-scored N comments
_PAGE_LIMIT = 500  # comments per page old.reddit will render

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT


class RedditFetchError(RuntimeError):
    """Raised when a thread cannot be fetched, with a human-readable reason."""


def fetch_thread(ref: ThreadRef, cap: int = COMMENT_CAP) -> list[Comment]:
    """Fetch up to `cap` top-scored comments for a Reddit thread.

    parent_id is preserved on every comment so the reply tree can be
    reconstructed downstream (a comment's parent may itself fall outside the
    top-`cap` set — that is expected under the guardrail).
    """
    if ref.platform != "reddit":
        raise RedditFetchError(f"reddit fetcher got a '{ref.platform}' thread reference.")

    thread_id = ref.thread_id or _resolve_share_link(ref.url)
    html = _get_html(
        f"https://old.reddit.com/comments/{thread_id}/",
        params={"sort": "top", "limit": _PAGE_LIMIT},
    )

    soup = BeautifulSoup(html, "html.parser")
    comment_area = soup.find("div", class_="commentarea")
    if not isinstance(comment_area, Tag):
        raise RedditFetchError(
            f"Reddit returned a page without a comment section for thread "
            f"'{thread_id}'. The thread may be deleted, private, or quarantined."
        )

    comments: list[Comment] = []
    forest = comment_area.find("div", class_="sitetable", recursive=False)
    if isinstance(forest, Tag):
        _walk_forest(forest, parent_id=None, comments=comments)

    comments.sort(key=lambda c: c.score, reverse=True)
    return comments[:cap]


def _walk_forest(sitetable: Tag, parent_id: str | None, comments: list[Comment]) -> None:
    """Depth-first walk of an old-reddit comment listing, in place.

    Reply structure comes from DOM nesting: each comment div holds its
    replies inside div.child > div.sitetable. "load more comments" stubs
    (data-type="morechildren") are skipped — with limit=500 on one page and
    a 200-comment cap they are almost never reached, and expanding them has
    no unauthenticated endpoint anymore.
    """
    for thing in sitetable.find_all("div", attrs={"data-type": "comment"}, recursive=False):
        comment = _to_comment(thing, parent_id)
        if comment is not None:
            comments.append(comment)
        child = thing.find("div", class_="child")
        if isinstance(child, Tag):
            replies = child.find("div", class_="sitetable", recursive=False)
            if isinstance(replies, Tag):
                child_parent = comment.id if comment else parent_id
                _walk_forest(replies, parent_id=child_parent, comments=comments)


def _to_comment(thing: Tag, parent_id: str | None) -> Comment | None:
    fullname = thing.get("data-fullname", "")
    if not isinstance(fullname, str) or not fullname.startswith("t1_"):
        return None
    entry = thing.find("div", class_="entry")
    if not isinstance(entry, Tag):
        return None

    author = thing.get("data-author") or "[deleted]"

    body = entry.find("div", class_="md")
    text = body.get_text(" ", strip=True) if isinstance(body, Tag) else ""

    score = 0
    score_span = entry.find("span", class_="unvoted")
    if isinstance(score_span, Tag):
        title = score_span.get("title")
        if isinstance(title, str) and title.lstrip("-").isdigit():
            score = int(title)
        else:  # e.g. "[score hidden]" or "1234 points"
            first = score_span.get_text(strip=True).split(" ")[0].replace(",", "")
            if first.lstrip("-").isdigit():
                score = int(first)

    timestamp = datetime.fromtimestamp(0, tz=timezone.utc)
    time_tag = entry.find("time")
    if isinstance(time_tag, Tag):
        raw = time_tag.get("datetime")
        if isinstance(raw, str):
            try:
                timestamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                pass

    return Comment(
        id=fullname[3:],
        author=str(author),
        text=text,
        parent_id=parent_id,
        timestamp=timestamp,
        score=score,
        platform="reddit",
    )


def _resolve_share_link(url: str) -> str:
    """Resolve a reddit.com/r/<sub>/s/<code> share link to its thread id."""
    from src.ingestion.detector import detect

    response = _session.get(url, allow_redirects=True, timeout=15)
    ref = detect(response.url)
    if ref.thread_id is None:
        raise RedditFetchError(
            f"Could not resolve the Reddit share link '{url}'. Open it in a "
            "browser and paste the full thread URL instead."
        )
    return ref.thread_id


def _get_html(url: str, params: dict[str, int | str]) -> str:
    try:
        response = _session.get(url, params=params, timeout=30)
    except requests.RequestException as exc:
        raise RedditFetchError(f"Network error while contacting Reddit: {exc}") from exc

    if response.status_code == 404:
        raise RedditFetchError(
            "Reddit says this thread does not exist (404). Check the URL — the "
            "thread may have been deleted."
        )
    if response.status_code == 403:
        raise RedditFetchError(
            "Reddit refused the request (403). The subreddit may be private or "
            "quarantined, or Reddit is blocking traffic from your network right "
            "now. Wait a minute and try again."
        )
    if response.status_code == 429:
        raise RedditFetchError(
            "Reddit rate limit hit (429). Wait a minute and try again — and "
            "note results are cached, so repeat runs on the same thread are free."
        )
    if not response.ok:
        raise RedditFetchError(f"Reddit returned HTTP {response.status_code} for {url}.")
    return response.text
