"""LangGraph wiring: dispatcher -> (parallel ingestion) -> enrich -> score -> alert.

The fan-out is the heart of this graph. Rather than chaining ingestion nodes
linearly (sec -> news -> screener), we use LangGraph's ``Send`` API so all
enabled sources run *simultaneously* in one superstep and then converge on the
enrichment node.

How the convergence works:
  * ``dispatcher`` is a trivial node that stamps the run timestamp.
  * ``_fan_out`` is a conditional-edge function that returns a list of ``Send``
    objects — one per enabled ingestion node. Returning a list is what triggers
    parallel execution.
  * Each ingestion node has a normal edge to ``enrich``. Because all the
    ingestion branches are active in the same superstep, LangGraph runs
    ``enrich`` (and then ``score``) exactly once, after every branch has
    finished, with the ``candidates``/``errors`` reducers having concatenated
    all the results.
  * ``enrich`` grounds the accumulated candidates with Perplexity (financials +
    cited research) before scoring. It is a no-op when no Perplexity key is set.
"""

from __future__ import annotations

from datetime import datetime, timezone

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .config import Config
from .nodes.alert import make_alert_node
from .nodes.enrich import make_enrich_node
from .nodes.ingest_news import make_ingest_news_node
from .nodes.ingest_screener import make_ingest_screener_node
from .nodes.ingest_sec import make_ingest_sec_node
from .nodes.score_blueprint import make_score_node
from .state import AgentState

# The ingestion nodes to fan out to. Add a source here and it's automatically
# included in the parallel fan-out (and the Send list below).
INGESTION_NODES = ("ingest_sec", "ingest_news", "ingest_screener")


def _dispatcher(state: AgentState) -> dict:
    """Entry node: stamp the run timestamp. The fan-out happens on its edges."""
    return {"run_timestamp": datetime.now(timezone.utc).isoformat()}


def _fan_out(state: AgentState):
    """Conditional edge: emit one Send per ingestion source for parallel run.

    Returning a *list* of Send objects is what makes LangGraph execute all the
    targets concurrently in the same superstep (true fan-out), instead of
    following a single static edge.
    """
    return [Send(node, state) for node in INGESTION_NODES]


def build_graph(config: Config):
    """Build and compile the agent graph, injecting ``config`` into each node."""
    graph = StateGraph(AgentState)

    # Nodes are built by factories so they close over config — config never has
    # to live in (and be serialized through) the graph state.
    graph.add_node("dispatcher", _dispatcher)
    graph.add_node("ingest_sec", make_ingest_sec_node(config))
    graph.add_node("ingest_news", make_ingest_news_node(config))
    graph.add_node("ingest_screener", make_ingest_screener_node(config))
    graph.add_node("enrich", make_enrich_node(config))
    graph.add_node("score", make_score_node(config))
    graph.add_node("alert", make_alert_node(config))

    graph.add_edge(START, "dispatcher")

    # Fan out from the dispatcher to all ingestion nodes in parallel. The third
    # arg lists possible Send targets so LangGraph can validate/visualize them.
    graph.add_conditional_edges("dispatcher", _fan_out, list(INGESTION_NODES))

    # Every ingestion branch converges on enrichment. enrich runs once, after
    # all branches complete, with accumulated candidates/errors; score then runs
    # once after enrich.
    for node in INGESTION_NODES:
        graph.add_edge(node, "enrich")

    graph.add_edge("enrich", "score")
    graph.add_edge("score", "alert")
    graph.add_edge("alert", END)

    return graph.compile()


def initial_state() -> AgentState:
    """A fresh, empty state for a run."""
    return AgentState(
        candidates=[],
        scored_candidates=[],
        alert_queue=[],
        run_timestamp="",
        errors=[],
    )
