"""LangGraph wiring: dispatcher -> (parallel ingestion) -> score -> alert.

The fan-out is the heart of this graph. Rather than chaining ingestion nodes
linearly (sec -> news -> screener), we use LangGraph's ``Send`` API so all
enabled sources run *simultaneously* in one superstep and then converge on the
scoring node.

How the convergence works:
  * ``dispatcher`` is a trivial node that stamps the run timestamp.
  * ``_fan_out`` is a conditional-edge function that returns a list of ``Send``
    objects — one per enabled ingestion node. Returning a list is what triggers
    parallel execution.
  * Each ingestion node has a normal edge to ``score``. Because all the
    ingestion branches are active in the same superstep, LangGraph runs
    ``score`` exactly once, after every branch has finished, with the
    ``candidates``/``errors`` reducers having concatenated all the results.
"""

from __future__ import annotations

from datetime import datetime, timezone

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .config import Config
from .nodes.alert import make_alert_node
from .nodes.classify_event import make_classify_event_node
from .nodes.critic import make_critic_node
from .nodes.decide import make_decide_node
from .nodes.extract_evidence import make_extract_evidence_node
from .nodes.hard_fail import make_hard_fail_node
from .nodes.ingest_news import make_ingest_news_node
from .nodes.ingest_screener import make_ingest_screener_node
from .nodes.ingest_sec import make_ingest_sec_node
from .nodes.normalize import make_normalize_node
from .nodes.output import make_output_node
from .nodes.score import make_score_node
from .nodes.synthesize import make_synthesize_node
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
    # v2 evidence-first linear chain.
    graph.add_node("normalize", make_normalize_node(config))
    graph.add_node("classify_event", make_classify_event_node(config))
    graph.add_node("extract_evidence", make_extract_evidence_node(config))
    graph.add_node("hard_fail", make_hard_fail_node(config))
    graph.add_node("score", make_score_node(config))
    graph.add_node("synthesize", make_synthesize_node(config))
    graph.add_node("critic", make_critic_node(config))
    graph.add_node("decide", make_decide_node(config))
    graph.add_node("output", make_output_node(config))
    graph.add_node("alert", make_alert_node(config))

    graph.add_edge(START, "dispatcher")

    # Fan out from the dispatcher to all ingestion nodes in parallel. The third
    # arg lists possible Send targets so LangGraph can validate/visualize them.
    graph.add_conditional_edges("dispatcher", _fan_out, list(INGESTION_NODES))

    # Every ingestion branch converges on normalize. It runs once, after all
    # branches complete, with accumulated candidates/errors. From there the
    # pipeline is a single linear chain (single-writer state, no reducers).
    for node in INGESTION_NODES:
        graph.add_edge(node, "normalize")

    graph.add_edge("normalize", "classify_event")
    graph.add_edge("classify_event", "extract_evidence")
    graph.add_edge("extract_evidence", "hard_fail")
    graph.add_edge("hard_fail", "score")
    graph.add_edge("score", "synthesize")
    graph.add_edge("synthesize", "critic")
    graph.add_edge("critic", "decide")
    graph.add_edge("decide", "output")
    graph.add_edge("output", "alert")
    graph.add_edge("alert", END)

    return graph.compile()


def initial_state() -> AgentState:
    """A fresh, empty state for a run."""
    return AgentState(
        candidates=[],
        normalized_candidates=[],
        classified_candidates=[],
        enriched_candidates=[],
        scored_candidates=[],
        decided_candidates=[],
        alert_queue=[],
        run_timestamp="",
        errors=[],
    )
