# JobFit

> A modular, observable LangChain assistant that analyzes job descriptions, tailors resumes, drafts grounded cover letters, and prepares interview questions — built incrementally to demonstrate the full LangChain & LangSmith stack.

**Status:** 🚧 In active development — V4 complete (production architecture, FastAPI, tests, CI)
---

## What this project is

A capstone project for the LangChain & LangSmith module of the CyberPro AI Developer Bootcamp. The assistant is built across six versions (V1–V6), each version adding one major capability and demonstrating one core concept:

| Version | Concept | Status |
| --- | --- | --- |
| V1 | LCEL fundamentals & structured outputs | ✅ Done |
| V2 | Routing & specialized handlers | ✅ Done |
| V3 | RAG with ChromaDB | ✅ Done |
| V4 | Production architecture & FastAPI | ✅ Done |
| V5 | RAG optimization & evaluation | ⏳ Not started |
| V6 | Controlled tools & action selection | ⏳ Not started |

📄 **See [`PROJECT_SPECIFICATION.md`](./PROJECT_SPECIFICATION.md) for the full design document.**

---

## Quick start

### Prerequisites

- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A [LangSmith API key](https://smith.langchain.com/settings) (free tier is fine)

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/IdanRodri17/JobFit.git
cd JobFit

# 2. Create and activate a virtual environment
python -m venv venv
source venv/Scripts/activate   # Git Bash on Windows
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and fill in your API keys
```

### Running (placeholder — coming in V1)

```bash
python main.py
```

---

## Tech stack

- **LangChain** (LCEL) for chain composition
- **LangSmith** for tracing and evaluation
- **OpenAI** (`gpt-4o-mini`) as the default model
- **ChromaDB** as the local vector store
- **Pydantic** for structured outputs
- **FastAPI** for the API layer (V4+)

---

## Author

Built by **Idan** as part of the CyberPro AI Developer Bootcamp (ELAD Software).

## License

Personal educational project. Not licensed for redistribution.
