# Argument Cartographer

Paste a Reddit thread URL, get back an interactive argument map: claims
extracted from comments, deduplicated semantically, connected by
supports / attacks / restates edges, with commenters clustered by stance.

**Status: Phase 3 (feature-complete v1)** — the full pipeline works and is
tested end to end: ingestion → Gemini claim extraction → semantic dedup →
supports/attacks/restates edge inference → stance clustering → interactive
pyvis map, all cached in SQLite.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then fill in GEMINI_API_KEY
```

## Run

```bash
.venv/bin/streamlit run app.py
```

Smoke-test the ingestion layer against any public Reddit thread:

```bash
.venv/bin/python scripts/smoke_test.py "https://www.reddit.com/r/AskReddit/comments/<id>/<slug>/"
```

Test extraction + dedup end to end (uses your GEMINI_API_KEY; add
`--max-comments 10` for a cheap dry run):

```bash
.venv/bin/python scripts/extraction_test.py "<thread_url>"
```

Test the graph layer (edge inference + stance clustering) on an
already-analyzed thread; writes an interactive HTML map to `data/`:

```bash
.venv/bin/python scripts/graph_test.py "<thread_url>"
```

## Claim extraction & dedup

`src/extraction/claims.py` sends comments to **gemini-2.5-flash** in batches
of 40 with a structured-output schema (pydantic-validated JSON), extracting
up to 3 atomic claims per comment. Pacing is tuned to the Gemini free tier
(5 requests/min): a 200-comment thread is 5 requests over ~1–3 minutes, with
patient retries on 429s. `src/dedup/pinecone_dedup.py` then embeds claims
locally (sentence-transformers, all-MiniLM-L6-v2) and merges paraphrases via
agglomerative clustering on cosine distance; each cluster keeps its medoid
as the canonical phrasing. Both results are cached in SQLite, so a thread is
only ever paid for once.

## Argument graph & stance clustering

`src/graph/edges.py` classifies relations between distinct claims with
gemini-2.5-flash. Rather than ask about all O(n²) claim pairs, it only sends
**candidate pairs**: claims from comments that actually replied to each
other, plus claims whose embeddings sit in a "same topic, not same claim"
similarity band. Each pair is labelled supports / attacks / restates / none
(none dropped). `src/graph/builder.py` assembles a `networkx.DiGraph` and
derives **stance groups** by running community detection over a
commenter-affinity graph (agreement from shared/supporting claims adds
weight, attacks subtract it). `src/graph/render.py` renders the result with
**pyvis**: nodes are claims (size = times asserted, color = asserters'
stance group), edges are green/red/gray-dashed by relation. Edges and
stances are cached in SQLite alongside the rest.

## How ingestion works (and why there's no Reddit API key)

Reddit gates API app creation behind manual approval
([Responsible Builder Policy](https://support.reddithelp.com/hc/en-us/articles/42728983564564-Responsible-Builder-Policy))
and deprecated the unauthenticated `.json` endpoints in May 2026. What
remains publicly served is old.reddit.com's server-rendered HTML, so
`src/ingestion/reddit.py` fetches a thread's old-reddit page
(`?sort=top&limit=500` — one request per thread) and parses the comment
tree out of the HTML with BeautifulSoup. Fetching is capped at the **top
200 comments by score** (v1 guardrail); `parent_id` is derived from DOM
nesting and preserved on every comment so the reply tree survives. Results
are cached in SQLite, so re-running a thread costs Reddit nothing. If
Reddit API approval lands later, a PRAW fetcher can replace this module
behind the same `fetch_thread(ThreadRef) -> list[Comment]` signature.

## Layout

```
app.py                        Streamlit entry point (URL → map + tabs)
src/models.py                 Pydantic models: Comment, Claim, ClaimCluster,
                              ClaimEdge, StanceGroup, ThreadRef
src/llm.py                    Shared Gemini client + throttled structured calls
src/ingestion/detector.py     URL validation/parsing; per-platform parser registry
src/ingestion/reddit.py       old.reddit HTML thread fetcher (no credentials)
src/cache.py                  SQLite cache keyed by URL hash (data/cache.db)
src/extraction/claims.py      Gemini (gemini-2.5-flash) claim extraction
src/dedup/pinecone_dedup.py   Semantic dedup (local embeddings + clustering)
src/graph/edges.py            Gemini supports/attacks/restates edge inference
src/graph/builder.py          networkx graph + commenter stance clustering
src/graph/render.py           Interactive pyvis HTML rendering
scripts/smoke_test.py         CLI: fetch a thread, print stats + first 10 comments
scripts/extraction_test.py    CLI: extraction + dedup stats on a real thread
scripts/graph_test.py         CLI: edge inference + stance groups + HTML map
```

Data crossing module boundaries is always pydantic models (`src/models.py`).
