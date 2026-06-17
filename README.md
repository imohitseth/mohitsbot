<div align="center">

# 🤖 Mohit's Bot — Personal Knowledge-Grounded AI Assistant

### A Retrieval-Augmented Conversational Agent That "Decodes" a Person, Not Just Documents

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.2-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.x-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-LLM-4285F4?style=flat-square&logo=google&logoColor=white)](https://ai.google.dev/)
[![Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-000000?style=flat-square&logo=vercel&logoColor=white)](https://vercel.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](#license)

[**🌐 Live Demo**](https://mohitsbot.vercel.app) · [**Report Bug**](https://github.com/imohitseth/mohitsbot/issues) · [**Request Feature**](https://github.com/imohitseth/mohitsbot/issues)

</div>

---

## 📖 Overview

**Mohit's Bot** is a lightweight, full-stack **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about its creator by grounding a large language model in a curated, private knowledge base — instead of letting the LLM hallucinate biographical facts.

The core engineering problem it solves is a common one in applied LLM systems: **how do you constrain a general-purpose foundation model to answer reliably and *only* from a specific, trusted source of truth, while still letting it behave naturally for everything outside that scope?**

The solution implemented here is a **context-injection pattern** — a static knowledge base and a strict system prompt are composed into every request sent to the Gemini model via LangChain, giving the bot:

- **Grounded answers** for any question about Mohit (background, education, skills, projects, social links)
- **Graceful, honest fallbacks** ("Looks like Mohit didn't feed any info on that!") instead of confident hallucination when the knowledge base has no answer
- **Unrestricted general-purpose chat** for anything unrelated to Mohit (e.g., factual or small-talk queries), so the bot doesn't feel artificially caged
- **Persona consistency** — a friendly, conversational tone enforced through deterministic prompt rules rather than fine-tuning

This is, in effect, a **minimal, single-document RAG pipeline** — the same architectural pattern that powers production support bots, internal knowledge assistants, and onboarding tools, scaled down to a clean, auditable, and fully serverless implementation.

---

## ✨ Features

**Conversational & UX**
- 💬 Real-time, single-page chat interface served directly by Flask (no separate frontend build step)
- 🧠 Persistent per-session conversation history exposed via a dedicated API endpoint
- 🎯 Context-aware responses — distinguishes "questions about Mohit" from "general knowledge" from "small talk," and routes behavior accordingly

**Engineering & System Design**
- 🧩 **Decoupled prompt and data layer** — `system_prompt.txt` (behavioral rules) and `knowledge_base.txt` (factual context) are external, swappable text files, not hard-coded strings — enabling persona/data iteration without touching application code
- 🔌 **Stateless, serverless backend** — Flask app packaged for Vercel's Python runtime, scaling to zero between requests
- 🔒 **Secrets isolated via environment variables** — the LLM API key is never committed to source control
- 🪶 **Minimal dependency footprint** — 6 pinned dependencies total, chosen deliberately over heavier alternatives (e.g., raw `langchain` over `langchain` + agents/tools overhead) to keep cold-start latency low on serverless infrastructure
- 🌍 **CORS-enabled API layer** — `flask-cors` allows the chat API to be safely consumed from a separate frontend origin (e.g., a personal portfolio site embedding the bot)

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.10+, HTML/CSS/JS | Backend logic + chat UI |
| **Backend Framework** | Flask 3.1.2 | Lightweight WSGI web server & REST API |
| **Cross-Origin Support** | flask-cors 6.0.1 | Secure cross-origin API access |
| **LLM Orchestration** | LangChain (`langchain-core`, `langchain-community`) | Prompt composition & model invocation pipeline |
| **LLM Provider** | Google Gemini (`langchain-google-genai`) | Natural language understanding & generation |
| **Knowledge Source** | Flat-file context store (`knowledge_base.txt`) | Ground-truth biographical data for retrieval/injection |
| **Config Management** | python-dotenv 1.1.1 | Environment-variable based secret management |
| **Hosting / Deployment** | Vercel (Python serverless runtime) | Zero-ops, auto-scaling deployment |
| **Frontend** | Jinja2 templates + static assets | Server-rendered chat widget UI |

> No traditional database is used — by design. The knowledge base is small, static, and read-only, so a flat file avoids the operational overhead (provisioning, connection pooling, migrations) a database would add for zero functional benefit at this scale.

---

## 🏗️ Architecture

```
┌──────────────────────┐         ┌──────────────────────────────────────────┐
│   Browser (Client)   │         │              Flask Server (api/)          │
│                       │  HTTP   │                                          │
│  templates/index.html│ ──────▶ │  GET  /              → render chat UI     │
│  + static/ (JS/CSS)   │         │  GET  /api/history   → return session log │
│                       │ ◀────── │  POST /api/chat      → handle a message   │
└──────────────────────┘  JSON   └──────────────┬───────────────────────────┘
                                                  │
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │   Context Assembly Layer        │
                                  │  system_prompt.txt (rules)      │
                                  │  + knowledge_base.txt (facts)   │
                                  │  + conversation history         │
                                  └───────────────┬─────────────────┘
                                                  │  LangChain message chain
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │   Google Gemini (via LangChain) │
                                  │   langchain-google-genai        │
                                  └───────────────┬─────────────────┘
                                                  │  generated response
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │   JSON response → client UI     │
                                  └───────────────────────────────┘
```

**Request flow:**
1. The browser loads the chat UI from Flask's `/` route (Jinja2-rendered `index.html`, styled via `static/`).
2. User messages are sent asynchronously to `POST /api/chat`.
3. The backend assembles a single LLM call composed of three layers: the **system prompt** (behavioral constraints), the **knowledge base** (factual grounding), and the **running conversation history** — this is the context-injection equivalent of a retrieval step, since the entire knowledge base is small enough to fit directly in-context rather than requiring vector search.
4. LangChain's Gemini integration sends this composed prompt to the model and returns a generated response.
5. The response is appended to session history and returned as JSON, which the frontend renders without a full page reload.
6. `GET /api/history` allows the client to rehydrate the visible conversation (e.g., on refresh) from server-held session state.

**Why this design:** at the current knowledge-base size, a full vector database (Pinecone/Chroma/FAISS) would add deployment complexity and latency with no retrieval-quality benefit — direct context injection is the *correct* engineering trade-off for the problem size, not a missing feature. The **Future Improvements** section below outlines the point at which that trade-off would flip.

---

## 📁 Folder Structure

```
mohitsbot/
├── api/                    # Flask application (Vercel serverless entry point)
│   └── app.py              # Routes, LangChain/Gemini wiring, session & chat logic
├── static/                 # CSS, JS, and image assets for the chat widget
│   └── logo.png            # Bot branding asset
├── templates/              # Jinja2 HTML templates
│   └── index.html          # Single-page chat interface
├── knowledge_base.txt      # External, swappable factual context (RAG source)
├── system_prompt.txt       # External, swappable persona & behavioral ruleset
├── requirements.txt        # Pinned Python dependencies
├── vercel.json              # Vercel build & routing configuration
└── README.md
```

**Why these matter:**
- **`api/app.py`** — the single source of backend truth. Vercel's config (`vercel.json`) builds this file with the `@vercel/python` runtime and routes *all* incoming traffic to it, making it both the API layer and the page-rendering layer.
- **`system_prompt.txt` / `knowledge_base.txt`** — deliberately kept outside the Python source. This separation means updating Mohit's bio or adjusting the bot's tone is a content change, not a code change — a small but real example of separating configuration from logic.
- **`vercel.json`** — declares a single build target (`api/app.py`) and a catch-all route, the minimal configuration needed to run a Flask app as a Vercel serverless function.

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+
- pip
- A [Google AI Studio](https://aistudio.google.com/) API key (for Gemini access)
- (Optional) [Vercel CLI](https://vercel.com/docs/cli) for production-style local testing

### 1. Clone the repository
```bash
git clone https://github.com/imohitseth/mohitsbot.git
cd mohitsbot
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file in the project root (see [Environment Variables](#-environment-variables) below).

### 5. Run locally
```bash
python api/app.py
```
The app will be available at `http://127.0.0.1:5000` (or the port Flask reports on startup).

### 6. Deploy to production (Vercel)
```bash
npm install -g vercel   # if not already installed
vercel login
vercel --prod
```
Vercel reads `vercel.json`, builds `api/app.py` with the Python runtime, and routes all paths to it automatically. Remember to add your environment variables in the **Vercel Project → Settings → Environment Variables** dashboard, since `.env` files are not deployed.

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | ✅ Yes | API key for Google Gemini, used by `langchain-google-genai` to authenticate model calls. Obtain from [Google AI Studio](https://aistudio.google.com/app/apikey). |
| `FLASK_SECRET_KEY` | Recommended | Secret key for signing Flask session cookies, used to persist per-user chat history securely. |
| `FLASK_ENV` / `FLASK_DEBUG` | Optional | Set to `development` / `1` for local debugging; should be unset or `production` in deployment. |

> ⚠️ Never commit `.env` to version control. Add it to `.gitignore` and configure secrets through your hosting provider's dashboard in production.

---

## 💡 Usage

Once running, the chat widget at `/` allows you to:

```text
You:  Who is Mohit?
Bot:  [Answers using knowledge_base.txt — education, skills, projects]

You:  What's the capital of France?
Bot:  [Answers directly — general knowledge, no knowledge-base lookup]

You:  Does Mohit know Rust?
Bot:  "Looks like Mohit didn't feed any info on that! Maybe you can ask him personally :)"

You:  Decode Mohit
Bot:  [Gives a short intro, then invites follow-up questions]
```

**Example workflows this design enables:**
- Embedding the bot as an interactive "About Me" section on a personal portfolio site
- Letting recruiters or visitors self-serve answers to common questions instead of re-reading a static résumé
- Demonstrating, live, how a constrained LLM persona behaves differently from an unconstrained one

---

## 🎯 Technical Highlights

This project demonstrates several practices directly relevant to backend and applied-AI engineering roles:

- **Prompt engineering as a first-class design artifact.** The system prompt isn't an afterthought — it's an explicit, version-controlled rule set with branching behavior (in-scope vs. out-of-scope vs. small talk vs. unknown-fact handling), effectively encoding a small decision tree in natural language rather than code.
- **Separation of concerns.** Persona (`system_prompt.txt`), data (`knowledge_base.txt`), orchestration (`app.py`), and presentation (`templates/`, `static/`) are cleanly partitioned — a structure that scales to a team workflow where a non-engineer could update content without a deploy.
- **API orchestration via LangChain.** Rather than calling the Gemini SDK directly, the project routes model calls through LangChain's abstraction layer, which decouples the application from any single LLM vendor — swapping providers later (e.g., to an OpenAI or Anthropic model) is a configuration change, not a rewrite.
- **Serverless-first architecture.** Designing for Vercel's Python runtime means embracing statelessness: no long-lived in-memory caches, no background threads, and idempotent request handling — the same constraints found in production cloud-function environments (AWS Lambda, Cloud Run, Cloud Functions).
- **Deliberate scope-limiting for reliability.** Explicitly defining what the bot *should not* claim to know (and what to say instead) is a concrete mitigation against LLM hallucination — a core reliability concern in any real-world LLM product.
- **Lean dependency management.** Six pinned dependencies keep the deployment artifact small, which directly reduces cold-start time on serverless infrastructure — a measurable performance consideration, not just tidiness.
- **Security-conscious secret handling.** API keys are sourced exclusively from environment variables via `python-dotenv`, keeping credentials out of source control and version history.

---

## 🧗 Challenges & Learnings

- **Constraining a general-purpose model without fine-tuning.** Fine-tuning Gemini wasn't necessary (or cost-effective) for a small, static fact set — the project shows that disciplined prompt engineering plus context injection can achieve reliable scoping behavior at a fraction of the cost and complexity.
- **Designing for graceful failure.** A naive chatbot either refuses everything outside its dataset or hallucinates confidently. Explicitly scripting the "I don't know" path (rather than letting the model improvise one) was the key insight that made the bot trustworthy rather than just impressive.
- **Choosing the right retrieval strategy for the problem size.** It would have been easy to over-engineer this with a vector database and embedding pipeline. Recognizing that direct context injection was the *appropriate* solution for a knowledge base this small — and reserving vector retrieval for when it's actually needed — was a deliberate architectural decision, not a shortcut.
- **Serverless constraints reshape backend habits.** Building for Vercel's stateless functions reinforced patterns (no persistent local state, fast cold starts, externalized configuration) that map directly to how production cloud-native services are built.

---

## 🚀 Future Improvements

- **Vector-based retrieval (RAG v2):** Migrate `knowledge_base.txt` into a vector store (e.g., FAISS or Pinecone) with embedding-based similarity search, enabling the knowledge base to scale beyond what fits comfortably in a single context window.
- **Streaming responses:** Use server-sent events or WebSockets to stream Gemini's output token-by-token instead of waiting for the full response, improving perceived latency.
- **Conversation persistence across sessions:** Replace in-memory/session-based history with a lightweight persistent store (e.g., SQLite or Redis) so chat history survives server restarts and cold starts.
- **Automated testing & CI:** Add unit tests for prompt-assembly logic and an integration test suite (e.g., GitHub Actions) to catch regressions in routing/response behavior before deployment.
- **Observability:** Add structured logging and basic usage analytics (e.g., query volume, fallback-rate tracking) to monitor how often the bot hits its "I don't know" path — a useful signal for knowledge-base gaps.
- **Rate limiting & abuse protection:** Add request throttling on `/api/chat` to protect the Gemini API quota from abuse on a publicly deployed endpoint.
- **Multi-turn memory summarization:** For longer conversations, summarize older turns instead of sending full history on every call, reducing token usage and cost.

---

## 📸 Screenshots / Demo

> _Add screenshots or a short GIF of the chat interface here._

```
[ Screenshot: Landing page — "Let's decode Mohit!" ]
[ Screenshot: Sample conversation — biographical Q&A ]
[ Screenshot: Fallback response for an out-of-scope question ]
[ GIF: End-to-end chat interaction ]
```

🔗 **Live deployment:** [mohitsbot.vercel.app](https://mohitsbot.vercel.app)

---

## 🏆 Resume-Worthy Impact

> Suggested bullet points for a résumé or LinkedIn project entry:

- Designed and deployed a **full-stack Retrieval-Augmented Generation (RAG) chatbot** using Flask, LangChain, and Google Gemini, serving live conversational queries through a serverless architecture on Vercel.
- Engineered a **context-injection pipeline** separating persona rules, factual knowledge, and conversation state into independently maintainable layers, reducing hallucination risk without model fine-tuning.
- Implemented **prompt-engineered scope control**, enabling the system to reliably distinguish in-domain, out-of-domain, and conversational queries and respond appropriately to each.
- Built a **stateless, serverless backend** optimized for cold-start performance through minimal, deliberately curated dependencies, deployed via CI-free Vercel Python runtime.
- Abstracted LLM provider integration through **LangChain's model-agnostic interface**, ensuring the system can switch underlying language models with configuration changes rather than code rewrites.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built by **[Mohit Seth](https://www.linkedin.com/in/imohitseth)** · [Portfolio](https://mohitseth.vercel.app) · [Instagram](https://www.instagram.com/myself_mohit_seth)

</div>
