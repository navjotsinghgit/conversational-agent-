# 🎬 AutoStream Conversational AI Agent

A Conversational AI Agent for **AutoStream** — a fictional SaaS platform providing automated video editing tools for content creators. The agent handles greetings, answers product/pricing/policy questions via RAG, and captures high-intent leads through a guided multi-turn conversation.

---

## 🚀 How to Run Locally

### 1. Prerequisites

- Python 3.9+
- A [Google AI Studio](https://aistudio.google.com/) API key (free tier works)

### 2. Clone & Install

```bash
git clone <repo-url>
cd conversational-agent-

# (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Set Your API Key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your-gemini-api-key-here
```

Or export it directly:

```bash
export GOOGLE_API_KEY="your-gemini-api-key-here"
```

### 4. Run the Agent

```bash
python main.py
```

You'll see the AutoStream welcome banner. Type your messages and press Enter. Type `quit` or `exit` to stop.

### Example Session

```
You: Hi there!
Agent: Hey! Welcome to AutoStream 🎬 ...

You: What are your pricing plans?
Agent: AutoStream offers two plans: Basic ($29/month) and Pro ($79/month) ...

You: I want to sign up for the Pro plan for my YouTube channel.
Agent: That's awesome! I'd love to get you started. Could I get your full name?

You: John Smith
Agent: Great, John! And what's your email address?

You: john@example.com
Agent: Perfect! Which creator platform are you primarily using?

You: YouTube
Agent: 🎉 You're all set, John! I've registered your interest ...

# Console output:
# ============================================================
#   ✅  Lead captured successfully!
#   Name    : John Smith
#   Email   : john@example.com
#   Platform: YouTube
# ============================================================
```

---

## 🏗️ Architecture Explanation (~200 words)

### Why LangGraph?

We chose **LangGraph** over AutoGen because it provides an explicit, auditable state machine model for multi-turn conversations. Each turn is a deterministic traversal of a graph — rather than AutoGen's agent-to-agent message passing — which makes intent routing, lead-collection gating, and debugging far simpler.

### How State Is Managed

The agent's shared state is an `AgentState` TypedDict holding:
- **`messages`** — the full conversation history, automatically merged by LangGraph's `add_messages` annotation.
- **`intent`** — the latest classified intent (`greeting | inquiry | high_intent`).
- **`lead_info`** — a dictionary accumulating `name`, `email`, and `platform` across turns.
- **`lead_captured`** / **`awaiting_lead`** — flags preventing premature or duplicate tool execution.

LangGraph's **MemorySaver checkpointer** persists this state across every graph invocation within a session via a unique `thread_id`. This gives us full 5–10+ turn memory with no manual history management.

### Graph Flow

```
User message → classify_intent → [router] → respond        (greeting / inquiry)
                                          → collect_lead   (high_intent / mid-collection)
                                                │
                                        mock_lead_capture()  ← only when name+email+platform present
```

The RAG pipeline uses **FAISS** + **Google Generative AI embeddings** over a local Markdown knowledge base, returning the top-4 most relevant chunks as context for the LLM.

---

## 📱 WhatsApp Deployment via Webhooks

### Overview

To deploy this agent on WhatsApp, you would use the **WhatsApp Business Cloud API** (Meta) combined with a webhook server:

### Architecture

```
WhatsApp User
     │  (sends message)
     ▼
WhatsApp Cloud API  ──►  Webhook Server (FastAPI / Flask)
                                │
                         Parse message body
                                │
                         Run LangGraph agent
                         (reuse thread_id = WhatsApp phone number
                          for per-user persistent memory)
                                │
                         Send reply via
                         WhatsApp Cloud API  POST /messages
                                │
                     ◄──────────┘
WhatsApp User receives reply
```

### Implementation Steps

1. **Create a Meta Developer App** at [developers.facebook.com](https://developers.facebook.com) and enable the WhatsApp Business Product.

2. **Build a Webhook Server** (e.g., FastAPI):

```python
from fastapi import FastAPI, Request
import httpx, os

app = FastAPI()

@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    data = await request.json()
    # Extract sender phone and message text
    entry = data["entry"][0]["changes"][0]["value"]
    phone = entry["messages"][0]["from"]
    text  = entry["messages"][0]["text"]["body"]

    # Use phone number as LangGraph thread_id → per-user memory
    reply = run_agent(user_input=text, thread_id=phone)

    # Send reply back via WhatsApp Cloud API
    async with httpx.AsyncClient() as client:
        await client.post(
            f"https://graph.facebook.com/v19.0/{os.environ['PHONE_NUMBER_ID']}/messages",
            headers={"Authorization": f"Bearer {os.environ['WHATSAPP_TOKEN']}"},
            json={"messaging_product": "whatsapp",
                  "to": phone, "type": "text",
                  "text": {"body": reply}},
        )
    return {"status": "ok"}
```

3. **Expose the webhook** publicly using [ngrok](https://ngrok.com) during development, or deploy to a cloud service (Railway, Render, etc.).

4. **Register the webhook URL** in the Meta Developer dashboard and subscribe to the `messages` webhook field.

5. **Map `thread_id` to the sender's phone number** — this gives every WhatsApp user their own isolated, persistent conversation memory automatically.

---

## 📁 Project Structure

```
conversational-agent-/
├── knowledge_base/
│   └── autostream_kb.md      # Local RAG knowledge base (pricing, features, policies)
├── agent/
│   ├── __init__.py
│   ├── state.py              # AgentState TypedDict
│   ├── rag.py                # FAISS-based RAG pipeline
│   ├── tools.py              # mock_lead_capture() tool
│   ├── nodes.py              # LangGraph node functions
│   └── graph.py              # LangGraph StateGraph + MemorySaver
├── main.py                   # CLI entry point
├── requirements.txt
└── README.md
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `langgraph` | State graph framework for multi-turn agent |
| `langchain` | LLM orchestration, document loaders, text splitters |
| `langchain-google-genai` | Gemini 1.5 Flash LLM + embeddings |
| `langchain-community` | FAISS vector store integration |
| `faiss-cpu` | Local vector similarity search |
| `unstructured` | Markdown document loading |
| `python-dotenv` | `.env` file support |

---

## 📄 License

MIT