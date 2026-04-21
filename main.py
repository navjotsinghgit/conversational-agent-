#!/usr/bin/env python3
"""
main.py — AutoStream Conversational AI Agent (CLI)
===================================================

Usage
-----
    export GOOGLE_API_KEY="your-gemini-api-key"
    python main.py

Type your messages and press Enter.  Type 'quit' or 'exit' to stop.
"""

import os
import sys
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# ── Load .env (if present) ────────────────────────────────────────────────────
load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    print(
        "❌  GOOGLE_API_KEY is not set.\n"
        "    Export it in your shell or create a .env file with:\n"
        "      GOOGLE_API_KEY=your-gemini-api-key\n"
    )
    sys.exit(1)

# ── Lazy import after env check ───────────────────────────────────────────────
from agent.graph import build_graph

BANNER = """
╔══════════════════════════════════════════════════════════╗
║         AutoStream AI Assistant  🎬                      ║
║  Powered by LangGraph + Gemini 1.5 Flash                 ║
║  Type 'quit' or 'exit' to end the session.               ║
╚══════════════════════════════════════════════════════════╝
"""


def run_cli():
    print(BANNER)

    # Build the LangGraph agent
    graph = build_graph()

    # Each CLI session gets a unique thread_id for isolated memory
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Seed the initial state
    initial_state = {
        "messages": [],
        "intent": "unknown",
        "lead_info": {},
        "lead_captured": False,
        "awaiting_lead": False,
    }

    # We stream the graph but only need the final state snapshot
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋  Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "bye"}:
            print("👋  Thanks for chatting with AutoStream! Have a great day!")
            break

        # Inject the new user message
        new_message = HumanMessage(content=user_input)

        # Run one graph turn.  `stream` with mode="values" yields the full
        # state after each node — we take the last one.
        final_state = None
        for chunk in graph.stream(
            {"messages": [new_message]},
            config=config,
            stream_mode="values",
        ):
            final_state = chunk

        if final_state is None:
            print("Agent: (no response)")
            continue

        # Print the last AI message
        messages = final_state.get("messages", [])
        for msg in reversed(messages):
            from langchain_core.messages import AIMessage
            if isinstance(msg, AIMessage):
                print(f"\nAgent: {msg.content}\n")
                break


if __name__ == "__main__":
    run_cli()
