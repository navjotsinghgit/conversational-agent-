"""
agent/state.py
Defines the shared AgentState TypedDict used across all LangGraph nodes.
"""

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    Shared state passed between LangGraph nodes.

    Fields:
        messages       : Full conversation history (auto-merged by LangGraph).
        intent         : Latest classified intent:
                            'greeting' | 'inquiry' | 'high_intent' | 'unknown'
        lead_info      : Partial / complete lead data collected so far.
                            Keys: 'name', 'email', 'platform'
        lead_captured  : True once mock_lead_capture() has been successfully called.
        awaiting_lead  : True while the agent is in the middle of collecting lead info.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    intent: str
    lead_info: dict          # e.g. {"name": "Alice", "email": "", "platform": ""}
    lead_captured: bool
    awaiting_lead: bool
