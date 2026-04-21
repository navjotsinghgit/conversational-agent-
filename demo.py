#!/usr/bin/env python3
"""
demo.py — AutoStream Agent Demo (Auto-Run)
==========================================
Automatically runs through the required demo conversation, showing:
  1. Agent answering a pricing question (RAG)
  2. Agent detecting high-intent
  3. Agent collecting user details (name → email → platform)
  4. Successful lead capture via mock_lead_capture()

Modes
-----
  Real mode   : set GOOGLE_API_KEY env var  → uses live Gemini 1.5 Flash
  Scripted mode: pass --mock flag            → uses pre-scripted responses
                 (guaranteed output, no API key needed)
"""

import os
import sys
import time
import textwrap

# ── ANSI colours ─────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
MAGENTA= "\033[95m"
WHITE  = "\033[97m"
BLUE   = "\033[94m"
RED    = "\033[91m"
BG_DARK= "\033[48;5;234m"

WIDTH  = 72


def hr(char="─", color=DIM):
    print(f"{color}{char * WIDTH}{RESET}")


def header():
    print()
    hr("═", CYAN)
    print(f"{CYAN}{BOLD}{'AutoStream AI — Live Demo':^{WIDTH}}{RESET}")
    print(f"{DIM}{'Powered by LangGraph + Gemini 1.5 Flash':^{WIDTH}}{RESET}")
    hr("═", CYAN)
    print()


def label(tag: str, color: str = YELLOW):
    print(f"\n{color}{BOLD}[{tag}]{RESET}")


def user_say(text: str, delay: float = 0.8):
    """Print the user message with a typing-effect delay."""
    time.sleep(delay)
    print(f"\n{YELLOW}You:{RESET}  {WHITE}{text}{RESET}")
    time.sleep(0.3)


def agent_say(text: str, delay: float = 0.6):
    """Print the agent response wrapped nicely."""
    time.sleep(delay)
    wrapped = textwrap.fill(text, width=WIDTH - 8)
    lines = wrapped.split("\n")
    print(f"\n{CYAN}{BOLD}Agent:{RESET}", end="")
    for i, line in enumerate(lines):
        if i == 0:
            print(f" {line}")
        else:
            print(f"       {line}")
    print()


def step_banner(n: int, title: str):
    print()
    hr()
    print(f"  {MAGENTA}{BOLD}STEP {n}{RESET}  {WHITE}{title}{RESET}")
    hr()


def tool_fire(name: str, email: str, platform: str):
    """Visual representation of the mock_lead_capture() tool call."""
    time.sleep(0.4)
    print()
    hr("═", GREEN)
    print(f"{GREEN}{BOLD}  🔧  TOOL EXECUTION: mock_lead_capture(){RESET}")
    hr("─", GREEN)
    print(f"{GREEN}  ✅  Lead captured successfully!{RESET}")
    print(f"  {BOLD}Name    :{RESET}  {WHITE}{name}{RESET}")
    print(f"  {BOLD}Email   :{RESET}  {WHITE}{email}{RESET}")
    print(f"  {BOLD}Platform:{RESET}  {WHITE}{platform}{RESET}")
    hr("═", GREEN)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MOCK SCRIPTED RESPONSES  (used when --mock flag or no API key)
# ─────────────────────────────────────────────────────────────────────────────

MOCK_CONVERSATION = [
    # (user_input, agent_response, is_tool_call)
    (
        "Hi there!",
        (
            "Hey! 👋 Welcome to AutoStream — your AI-powered video editing "
            "companion for content creators! I'm here to help you explore our "
            "plans, features, and get you set up. What can I help you with today?"
        ),
        False,
    ),
    (
        "What are your pricing plans?",
        (
            "Great question! AutoStream offers two plans:\n\n"
            "  📦  Basic Plan — $29/month\n"
            "      • 10 videos/month  •  720p resolution\n"
            "      • 50+ templates  •  Email support (Mon–Fri)\n\n"
            "  🚀  Pro Plan — $79/month\n"
            "      • Unlimited videos  •  4K resolution\n"
            "      • AI captions (multi-language)  •  500+ templates\n"
            "      • 24/7 priority support  •  Team collaboration\n\n"
            "Both plans come with a 7-day free trial. Would you like to know more "
            "about a specific plan or feature?"
        ),
        False,
    ),
    (
        "What's the refund policy?",
        (
            "Our refund policy is straightforward: you're eligible for a full "
            "refund within 7 days of your purchase. After that 7-day window, "
            "refunds are not available. If approved, refunds are processed within "
            "5–7 business days. Also note that if you cancel your subscription, "
            "it remains active until the end of your current billing period — "
            "no partial refunds for unused days."
        ),
        False,
    ),
    (
        "That sounds great — I want to sign up for the Pro plan for my YouTube channel!",
        (
            "That's fantastic — the Pro plan is a perfect fit for YouTube creators! 🎬 "
            "Let me get you registered. Could I start with your full name?"
        ),
        False,
    ),
    (
        "John Smith",
        (
            "Great to meet you, John! 😊 And what email address should we use "
            "to send your account details and onboarding info?"
        ),
        False,
    ),
    (
        "john.smith@gmail.com",
        (
            "Perfect! Last question — you mentioned YouTube, but just to confirm: "
            "which creator platform are you primarily creating content for? "
            "(e.g. YouTube, Instagram, TikTok)"
        ),
        False,
    ),
    (
        "YouTube",
        (
            "🎉 You're all set, John! I've registered your interest in the "
            "AutoStream Pro plan. Our team will reach out to john.smith@gmail.com "
            "shortly with your personalised onboarding for YouTube creators. "
            "Welcome aboard! 🚀"
        ),
        True,  # triggers tool display
    ),
]


def run_mock_demo():
    """Run the fully scripted demo (no API key required)."""
    header()

    print(f"{DIM}  Mode: Scripted Demo  |  Framework: LangGraph + Gemini 1.5 Flash{RESET}")
    print(f"{DIM}  Knowledge Base: knowledge_base/autostream_kb.md (FAISS RAG){RESET}")
    time.sleep(1.5)

    step_titles = [
        None,
        None,
        None,
        "Intent: Greeting  →  Node: respond",
        "Intent: Inquiry   →  Node: respond (RAG retrieval)",
        "Intent: Inquiry   →  Node: respond (RAG retrieval)",
        "Intent: HIGH_INTENT  →  Node: collect_lead",
        "Collecting lead — field: email",
        "Collecting lead — field: platform",
        "All fields collected  →  mock_lead_capture() fired ✅",
    ]

    step_map = {0: 1, 1: 2, 3: 3}  # conversation index → step number

    steps = [
        (0, 1, "Greeting — Agent warms up"),
        (1, 2, "Pricing Question — RAG retrieval from knowledge base"),
        (2, 2, "Policy Question — RAG retrieval from knowledge base"),
        (3, 3, "High-Intent Detected — Lead collection begins"),
        (4, 3, "Lead Collection — Name captured, asking for email"),
        (5, 3, "Lead Collection — Email captured, asking for platform"),
        (6, 4, "All details collected  →  mock_lead_capture() fires!"),
    ]

    for conv_idx, step_num, step_title in steps:
        step_banner(step_num, step_title)
        user_input, agent_response, is_tool = MOCK_CONVERSATION[conv_idx]

        # Show intent analysis for interesting turns
        if conv_idx == 1:
            print(f"  {DIM}🔍 Intent classified: {CYAN}inquiry{DIM} → routing to {CYAN}respond node{DIM} (RAG){RESET}")
            print(f"  {DIM}📚 Retrieving from FAISS index... top-4 chunks returned{RESET}")
            time.sleep(0.5)
        elif conv_idx == 3:
            print(f"  {DIM}🔍 Intent classified: {RED}{BOLD}HIGH_INTENT{DIM} → routing to {RED}collect_lead node{RESET}")
            time.sleep(0.5)
        elif conv_idx in (4, 5):
            print(f"  {DIM}🔍 awaiting_lead=True → routing to {RED}collect_lead node{RESET}")
            time.sleep(0.5)
        elif conv_idx == 6:
            print(f"  {DIM}🔍 All fields present: name ✓  email ✓  platform ✓{RESET}")
            time.sleep(0.5)

        user_say(user_input)
        agent_say(agent_response)

        if is_tool:
            tool_fire("John Smith", "john.smith@gmail.com", "YouTube")

        time.sleep(0.5)

    hr("═", CYAN)
    print(f"{GREEN}{BOLD}{'  Demo Complete — All 4 capabilities demonstrated!':^{WIDTH}}{RESET}")
    hr("═", CYAN)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# LIVE MODE  (uses real Gemini 1.5 Flash + FAISS RAG)
# ─────────────────────────────────────────────────────────────────────────────

LIVE_INPUTS = [
    "Hi there!",
    "What are your pricing plans?",
    "What's the refund policy?",
    "That sounds great — I want to sign up for the Pro plan for my YouTube channel!",
    "John Smith",
    "john.smith@gmail.com",
    "YouTube",
]

LIVE_STEP_MAP = [
    (1, "Greeting — Agent responds"),
    (2, "Pricing Question — RAG retrieval"),
    (2, "Refund Policy — RAG retrieval"),
    (3, "HIGH_INTENT detected — Lead collection starts"),
    (3, "Lead collection — name captured"),
    (3, "Lead collection — email captured"),
    (4, "All details collected — mock_lead_capture() fires!"),
]


def run_live_demo():
    """Run demo with real Gemini API + LangGraph."""
    from dotenv import load_dotenv
    load_dotenv()

    from langchain_core.messages import HumanMessage, AIMessage
    from agent.graph import build_graph
    import uuid

    header()
    print(f"{DIM}  Mode: LIVE (Gemini 1.5 Flash)  |  Framework: LangGraph + MemorySaver{RESET}")
    print(f"{DIM}  Knowledge Base: knowledge_base/autostream_kb.md (FAISS RAG){RESET}")
    print(f"{DIM}  Building FAISS index from knowledge base...{RESET}", end="", flush=True)

    graph = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f" {GREEN}done{RESET}")
    time.sleep(1)

    for i, user_input in enumerate(LIVE_INPUTS):
        step_num, step_title = LIVE_STEP_MAP[i]
        step_banner(step_num, step_title)
        user_say(user_input, delay=1.0)

        print(f"  {DIM}⏳ Thinking...{RESET}", end="", flush=True)

        final_state = None
        for chunk in graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="values",
        ):
            final_state = chunk

        # Clear "Thinking..." line
        print(f"\r{' ' * 20}\r", end="")

        if final_state:
            messages = final_state.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    agent_say(msg.content, delay=0.2)
                    break

            # Show tool fire if lead was just captured
            lead_info = final_state.get("lead_info", {})
            if final_state.get("lead_captured") and i == len(LIVE_INPUTS) - 1:
                tool_fire(
                    lead_info.get("name", ""),
                    lead_info.get("email", ""),
                    lead_info.get("platform", ""),
                )

        time.sleep(0.5)

    hr("═", CYAN)
    print(f"{GREEN}{BOLD}{'  Demo Complete — All 4 capabilities demonstrated!':^{WIDTH}}{RESET}")
    hr("═", CYAN)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    use_mock = "--mock" in sys.argv or not os.environ.get("GOOGLE_API_KEY")

    if use_mock:
        if not os.environ.get("GOOGLE_API_KEY"):
            print(f"\n{YELLOW}⚠  No GOOGLE_API_KEY found — running in scripted demo mode.{RESET}")
            print(f"{DIM}   Set GOOGLE_API_KEY to run with live Gemini 1.5 Flash.{RESET}\n")
            time.sleep(1.5)
        run_mock_demo()
    else:
        print(f"\n{GREEN}✅  GOOGLE_API_KEY detected — running live with Gemini 1.5 Flash.{RESET}\n")
        time.sleep(1)
        run_live_demo()
