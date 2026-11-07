"""Argument Cartographer — Streamlit entry point.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from src import cache
from src.dedup.pinecone_dedup import dedupe_claims
from src.extraction.claims import extract_claims
from src.graph.builder import build_graph, cluster_stances
from src.graph.edges import infer_edges
from src.graph.render import graph_to_html
from src.ingestion.detector import UnsupportedURLError, detect
from src.ingestion.reddit import COMMENT_CAP, RedditFetchError, fetch_thread
from src.llm import LLMError

load_dotenv()

CACHE_MAX_AGE_S = 60 * 60  # re-fetch threads after an hour

st.set_page_config(page_title="Argument Cartographer", page_icon="🗺️", layout="wide")
st.title("🗺️ Argument Cartographer")
st.caption(
    "Paste a Reddit thread URL to map its argument structure. "
    f"v1 ingests the top {COMMENT_CAP} comments by score."
)

url = st.text_input(
    "Thread URL",
    placeholder="https://www.reddit.com/r/AskReddit/comments/.../...",
)

if st.button("Map this thread", type="primary") and url:
    try:
        ref = detect(url)
    except UnsupportedURLError as exc:
        st.error(str(exc))
        st.stop()

    # --- Ingestion (cached) ---
    comments = cache.get_comments(url, max_age_s=CACHE_MAX_AGE_S)
    if comments is None:
        with st.spinner(f"Fetching thread from {ref.platform}…"):
            try:
                comments = fetch_thread(ref)
            except RedditFetchError as exc:
                st.error(str(exc))
                st.stop()
        cache.set_comments(url, comments)

    # --- Claim extraction + dedup (cached) ---
    analysis = cache.get_analysis(url)
    if analysis is None:
        progress = st.progress(0.0, text="Extracting claims with Gemini…")
        try:
            claims = extract_claims(
                comments,
                on_progress=lambda done, total: progress.progress(
                    done / total, text=f"Extracting claims with Gemini… {done}/{total} batches"
                ),
            )
        except LLMError as exc:
            st.error(str(exc))
            st.stop()
        progress.progress(1.0, text="Deduplicating claims…")
        clusters = dedupe_claims(claims)
        cache.set_analysis(url, claims, clusters)
        progress.empty()
    else:
        claims, clusters = analysis

    # --- Edge inference + stance clustering (cached) ---
    graph_data = cache.get_graph(url)
    if graph_data is None:
        progress = st.progress(0.0, text="Classifying claim relations with Gemini…")
        try:
            edges = infer_edges(
                clusters,
                comments,
                on_progress=lambda done, total: progress.progress(
                    done / total,
                    text=f"Classifying claim relations… {done}/{total} batches",
                ),
            )
        except LLMError as exc:
            st.error(str(exc))
            st.stop()
        stances = cluster_stances(clusters, edges, comments)
        cache.set_graph(url, edges, stances)
        progress.empty()
    else:
        edges, stances = graph_data

    graph = build_graph(clusters, edges, comments)

    # --- Results ---
    by_comment_id = {c.id: c for c in comments}
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Comments", len(comments))
    m2.metric("Commenters", len({c.author for c in comments}))
    m3.metric("Distinct claims", len(clusters))
    m4.metric("Relations", len(edges))
    m5.metric("Stance groups", len(stances))

    tab_map, tab_clusters, tab_stances, tab_claims, tab_comments = st.tabs(
        ["Argument map", "Distinct claims", "Stance groups", "All claims", "Comments"]
    )

    with tab_map:
        if edges:
            st.caption(
                "Nodes are distinct claims (size = how often asserted; color = "
                "asserters' stance group). Edges: green = supports, "
                "red = attacks, gray dashed = restates. Drag to explore."
            )
            connected = {n for e in edges for n in (e.source, e.target)}
            components.html(
                graph_to_html(graph.subgraph(connected), stances),
                height=660,
                scrolling=False,
            )
            st.caption(
                f"Showing the {len(connected)} claims that participate in at "
                f"least one relation (of {len(clusters)} total)."
            )
        else:
            st.info(
                "No argumentative relations found between claims — this thread "
                "may be a collection of independent anecdotes rather than a debate."
            )

    with tab_clusters:
        for cluster in clusters:
            n = len(cluster.claim_ids)
            authors = {
                by_comment_id[cid].author
                for cid in cluster.comment_ids
                if cid in by_comment_id
            }
            with st.expander(f"({n}×) {cluster.text}", expanded=False):
                st.write(f"Asserted by: {', '.join(sorted(authors)) or '?'}")

    with tab_stances:
        if stances:
            for group in stances:
                st.markdown(
                    f"**{group.label}** ({len(group.authors)} commenters): "
                    + ", ".join(f"u/{a}" for a in group.authors)
                )
        else:
            st.info("No stance groups detected (not enough interaction between commenters).")

    with tab_claims:
        st.dataframe(
            pd.DataFrame([c.model_dump() for c in claims]),
            width="stretch",
            hide_index=True,
        )

    with tab_comments:
        st.dataframe(
            pd.DataFrame([c.model_dump() for c in comments]),
            width="stretch",
            hide_index=True,
        )
