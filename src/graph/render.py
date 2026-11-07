"""Render the argument graph to interactive HTML with pyvis."""
from __future__ import annotations

import textwrap

import networkx as nx
from pyvis.network import Network

from src.models import StanceGroup

_EDGE_STYLE = {
    "supports": {"color": "#2e9e5b", "dashes": False},
    "attacks": {"color": "#d64545", "dashes": False},
    "restates": {"color": "#8a8f98", "dashes": True},
}
_GROUP_COLORS = ["#4c78d9", "#e08b3d", "#8e5bd1", "#3aa6a6", "#c95f8e", "#7a934d"]
_NEUTRAL_COLOR = "#9aa4b2"


def graph_to_html(graph: nx.DiGraph, stances: list[StanceGroup], height: str = "640px") -> str:
    """Return a self-contained HTML page of the interactive argument map.

    Node size tracks how many commenters assert the claim; node color is the
    stance group most of its asserters belong to (gray = unaligned).
    """
    group_of_author = {
        author: n for n, group in enumerate(stances) for author in group.authors
    }

    net = Network(height=height, width="100%", directed=True, cdn_resources="in_line")
    for node, data in graph.nodes(data=True):
        label = textwrap.shorten(data["text"], width=60, placeholder="…")
        tooltip = (
            f"{data['text']}\n\nAsserted in {data['assertions']} claim(s) by: "
            + ", ".join(data["authors"])
        )
        net.add_node(
            node,
            label="\n".join(textwrap.wrap(label, 24)),
            title=tooltip,
            size=12 + 4 * data["assertions"],
            color=_node_color(data["authors"], group_of_author),
            shape="dot",
        )
    for source, target, data in graph.edges(data=True):
        style = _EDGE_STYLE[data["relation"]]
        net.add_edge(
            source,
            target,
            title=data["relation"],
            color=style["color"],
            dashes=style["dashes"],
            arrows="to",
            width=2,
        )

    net.barnes_hut(gravity=-6000, spring_length=160, overlap=0.2)
    return net.generate_html(notebook=False)


def _node_color(authors: list[str], group_of_author: dict[str, int]) -> str:
    """Majority stance group of the claim's asserters; gray when unaligned."""
    votes: dict[int, int] = {}
    for author in authors:
        group = group_of_author.get(author)
        if group is not None:
            votes[group] = votes.get(group, 0) + 1
    if not votes:
        return _NEUTRAL_COLOR
    best = max(votes, key=lambda g: votes[g])
    return _GROUP_COLORS[best % len(_GROUP_COLORS)]
