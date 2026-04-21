"""
agent/nodes.py
LangGraph node functions.

Nodes
-----
classify_intent   – Labels the latest user message as greeting / inquiry / high_intent.
respond           – Generates a reply (greeting or RAG-augmented answer).
collect_lead      – Gathers name / email / platform one field at a time;
                    calls mock_lead_capture() only when all three are present.
router            – Pure routing logic (no LLM call); decides which node runs next.
"""

import os
import re
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from .state import AgentState
from .rag import retrieve
from .tools import mock_lead_capture

# ─── LLM singleton ────────────────────────────────────────────────────────────

def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.3,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  CLASSIFY INTENT
# ═══════════════════════════════════════════════════════════════════════════════

_INTENT_SYSTEM = """You are an intent classifier for AutoStream, a SaaS video editing company.

Classify the LAST user message into EXACTLY ONE of these labels:
  greeting   – A simple hello / hi / how are you / introductory message.
  inquiry    – A question about features, pricing, plans, policies, or the product.
  high_intent – The user clearly wants to sign up, start a trial, purchase, or buy.

Respond with ONLY the label — no punctuation, no explanation."""


def classify_intent(state: AgentState) -> AgentState:
    """Classify the latest user message and update state['intent']."""
    llm = _get_llm()
    last_user_msg = _last_human_message(state)

    response = llm.invoke(
        [
            SystemMessage(content=_INTENT_SYSTEM),
            HumanMessage(content=last_user_msg),
        ]
    )
    raw = response.content.strip().lower()

    # Accept partial matches to be robust against minor LLM deviations
    if "high" in raw or "intent" in raw or "sign" in raw or "buy" in raw:
        intent = "high_intent"
    elif "greet" in raw or "hello" in raw:
        intent = "greeting"
    else:
        intent = "inquiry"

    return {**state, "intent": intent}


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  RESPOND  (greeting or RAG-powered answer)
# ═══════════════════════════════════════════════════════════════════════════════

_GREETING_SYSTEM = """You are a friendly, enthusiastic sales assistant for AutoStream —
an AI-powered video editing SaaS for content creators.
Greet the user warmly in 2–3 sentences. Mention that you can help with pricing,
features, and getting started."""

_RAG_SYSTEM = """You are a helpful support and sales assistant for AutoStream,
an AI-powered video editing SaaS.

Use ONLY the knowledge base context provided below to answer the user's question.
Be concise, accurate, and friendly. If the answer is not in the context, say you
don't have that information and suggest they contact support@autostream.io.

--- KNOWLEDGE BASE CONTEXT ---
{context}
--- END CONTEXT ---"""


def respond(state: AgentState) -> AgentState:
    """Generate a reply for greeting or product/policy inquiry intents."""
    llm = _get_llm()
    last_user_msg = _last_human_message(state)
    intent = state.get("intent", "inquiry")

    if intent == "greeting":
        system = _GREETING_SYSTEM
        messages_to_send = [SystemMessage(content=system)] + list(state["messages"])
    else:
        context = retrieve(last_user_msg)
        system = _RAG_SYSTEM.format(context=context)
        messages_to_send = [SystemMessage(content=system)] + list(state["messages"])

    response = llm.invoke(messages_to_send)
    return {**state, "messages": [AIMessage(content=response.content)]}


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  COLLECT LEAD
# ═══════════════════════════════════════════════════════════════════════════════

_LEAD_SYSTEM = """You are a warm, professional sales assistant for AutoStream.
The user wants to sign up. Your job is to collect three pieces of information
ONE AT A TIME in this order:
  1. Full name
  2. Email address
  3. Creator platform (e.g. YouTube, Instagram, TikTok)

Rules:
- Ask for only ONE missing field per message.
- Be warm and conversational, not robotic.
- If the user gives multiple pieces of info at once, accept them all.
- Once you have all three, confirm enthusiastically and say you're completing registration.

Current lead info: {lead_info}
Next field needed: {next_field}
"""

_EXTRACT_SYSTEM = """Extract structured information from the user message.
Return ONLY a JSON object with any of these keys that you can find:
  name, email, platform
If a value is not present, omit that key.
Example: {{"name": "Alice Nguyen", "email": "alice@example.com"}}
Do NOT include any explanation, only the JSON object."""


def collect_lead(state: AgentState) -> AgentState:
    """
    Ask for missing lead fields one at a time.
    Calls mock_lead_capture() when all three (name, email, platform) are present.
    """
    llm = _get_llm()
    last_user_msg = _last_human_message(state)
    lead_info: dict = dict(state.get("lead_info") or {})

    # ── Step 1: Try to extract info from the latest user message ─────────────
    extract_response = llm.invoke(
        [
            SystemMessage(content=_EXTRACT_SYSTEM),
            HumanMessage(content=last_user_msg),
        ]
    )
    extracted = _safe_parse_json(extract_response.content)

    # Merge extracted fields (don't overwrite already-captured values)
    for field in ("name", "email", "platform"):
        if field not in lead_info and field in extracted:
            lead_info[field] = extracted[field]

    # ── Step 2: Check if all fields collected ─────────────────────────────────
    missing = [f for f in ("name", "email", "platform") if not lead_info.get(f)]

    if not missing:
        # All fields collected → fire the tool
        result = mock_lead_capture(
            name=lead_info["name"],
            email=lead_info["email"],
            platform=lead_info["platform"],
        )
        reply = (
            f"🎉 You're all set, **{lead_info['name']}**! "
            f"I've registered your interest in the AutoStream Pro plan. "
            f"Our team will reach out to **{lead_info['email']}** shortly with "
            f"your personalised onboarding for {lead_info['platform']} creators. "
            f"Welcome aboard! 🚀"
        )
        return {
            **state,
            "messages": [AIMessage(content=reply)],
            "lead_info": lead_info,
            "lead_captured": True,
            "awaiting_lead": False,
        }

    # ── Step 3: Ask for the next missing field ────────────────────────────────
    next_field = missing[0]
    system = _LEAD_SYSTEM.format(
        lead_info={k: v for k, v in lead_info.items() if v},
        next_field=next_field,
    )
    messages_to_send = [SystemMessage(content=system)] + list(state["messages"])
    response = llm.invoke(messages_to_send)

    return {
        **state,
        "messages": [AIMessage(content=response.content)],
        "lead_info": lead_info,
        "awaiting_lead": True,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTER  (no LLM call — pure logic)
# ═══════════════════════════════════════════════════════════════════════════════

def router(state: AgentState) -> str:
    """
    Decide which node to run next.

    Return value is the edge label used in the LangGraph conditional edges.
    """
    # If we are mid lead-collection and not yet done, stay in collect_lead
    if state.get("awaiting_lead") and not state.get("lead_captured"):
        return "collect_lead"

    intent = state.get("intent", "inquiry")
    if intent == "high_intent":
        return "collect_lead"
    return "respond"


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _last_human_message(state: AgentState) -> str:
    """Return the content of the most recent HumanMessage in state."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _safe_parse_json(text: str) -> dict:
    """Parse a JSON object from the LLM response, ignoring surrounding noise."""
    import json
    # Find the first '{...}' block
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {}
