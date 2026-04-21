"""
agent/graph.py
Constructs and compiles the LangGraph StateGraph for the AutoStream agent.

Graph topology
--------------

  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │   START ──► classify_intent ──► [router] ──► respond       │
  │                                           └──► collect_lead │
  │                                                    │        │
  │                                                    ▼        │
  │                                                   END       │
  └─────────────────────────────────────────────────────────────┘

Memory / persistence
--------------------
We use LangGraph's MemorySaver checkpointer so that the full conversation
history (messages, intent, lead_info, etc.) is retained across turns within the
same `thread_id`.  Each CLI session generates one thread_id.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .nodes import classify_intent, respond, collect_lead, router


def build_graph():
    """Build, compile, and return the LangGraph agent."""

    # ── Graph definition ──────────────────────────────────────────────────────
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("respond", respond)
    builder.add_node("collect_lead", collect_lead)

    # ── Edges ─────────────────────────────────────────────────────────────────
    # Entry point
    builder.set_entry_point("classify_intent")

    # After classification, route conditionally
    builder.add_conditional_edges(
        "classify_intent",
        router,
        {
            "respond": "respond",
            "collect_lead": "collect_lead",
        },
    )

    # Both terminal nodes end the graph turn
    builder.add_edge("respond", END)
    builder.add_edge("collect_lead", END)

    # ── Checkpointer (in-memory persistence) ──────────────────────────────────
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    return graph
