# 🤖 TestRAGic - Turning Inputs into Test Cases — Like Magic, but Smarter

> **An intelligent, end-to-end QA automation platform that revolutionizes software testing through AI-powered test generation, automated execution, and comprehensive analytics.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [Getting Started](#getting-started)
- [Use Cases](#use-cases)
- [Current Status](#current-status)
- [Future Roadmap](#future-roadmap)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 📖 Overview

**TestRAGic** combines **AI-powered test generation**, **automated test execution**, and **comprehensive reporting** into a unified, intelligent QA automation platform. Transform videos, documents, and requirements into executable test cases in minutes.

### What Makes TestRAGic Different?

**AI-Driven Automation** - Uses RAG + LLM for intelligent test case generation  
 **Multi-Modal Input** - Videos, YouTube links, documents, and mock data  
 **Parallel Execution** - Run tests simultaneously across browsers  
 **Smart Analytics** - AI-powered failure analysis and trend tracking  
 **End-to-End Platform** - Generation → Execution → Reporting in one place

---

## 🚀 Key Features

### 1. Intelligent Test Case Generation

| Feature                 | Description                                         |
| ----------------------- | --------------------------------------------------- |
| **Video-to-Tests**      | Upload demo videos → Automatic test case generation |
| **YouTube Integration** | Process how-to videos directly from YouTube URLs    |
| **Document Processing** | Convert requirements into executable test cases     |
| **RAG + LLM**           | Context-aware test generation with vector retrieval |

### 2. Automated Test Execution

| Feature                | Description                                    |
| ---------------------- | ---------------------------------------------- |
| **Multi-Browser**      | Chrome, Firefox, Safari with one configuration |
| **Cross-Platform**     | Desktop and mobile device testing              |
| **Parallel Execution** | Run multiple tests simultaneously for speed    |
| **Rich Artifacts**     | Screenshots, videos, and traces on failures    |

### 3. Comprehensive Analytics & Reporting

| Feature                  | Description                                 |
| ------------------------ | ------------------------------------------- |
| **Real-time Dashboards** | Live execution monitoring and status        |
| **Trend Analysis**       | Historical performance tracking and metrics |
| **Multi-Format Reports** | JSON, CSV, HTML, and PDF exports            |
| **Failure Analysis**     | AI-powered root cause insights              |

### 4. Multi-Modal Input Support

- **Video Files**: MP4, AVI, MOV, MKV, WebM
- **YouTube URLs**: Direct link processing with fallbacks
- **Mock Data**: Built-in test data for demos and development

---

## 🏗️ System Architecture

### Core Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: DATA INGESTION                                    │
│ • Video/YouTube upload → Transcript extraction             │
│ • Content processing → Visual element analysis             │
│ • Chunk segmentation → Logical test scenario identification│
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: TEST GENERATION                                   │
│ • RAG pipeline → Vector database retrieval                 │
│ • LLM analysis → Test scenario generation                  │
│ • Test case creation → Structured output (JSON, Markdown)  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: TEST EXECUTION                                    │
│ • Playwright conversion → Browser automation               │
│ • Multi-browser/device execution → Real-time monitoring    │
│ • Artifact collection → Logs, screenshots, videos          │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 4: RESULTS & REPORTING                               │
│ • Analytics dashboard → Visual test results                │
│ • Trend analysis → Historical metrics                      │
│ • Report generation → Stakeholder reports                  │
│ • Failure insights → AI-powered recommendations            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### Backend & Core

| Layer                  | Technologies                           |
| ---------------------- | -------------------------------------- |
| **Language**           | Python 3.13                            |
| **LLM Gateway**        | OmniRoute (350+ providers), OpenAI SDK |
| **Browser Automation** | Playwright                             |
| **Vector Database**    | FAISS (`IndexFlatL2`)                  |
| **Video Processing**   | PyTube, yt-dlp, YouTube Transcript API |
| **Speech-to-Text**     | Whisper (fallback), YouTube API        |

### Frontend & UI

| Component           | Technology |
| ------------------- | ---------- |
| **Web Dashboard**   | Streamlit  |
| **Visualizations**  | Plotly     |
| **Data Processing** | Pandas     |
| **Styling**         | Custom CSS |

### AI & ML Components

- **Large Language Models**: any provider reachable through OmniRoute; `auto` by default
- **RAG Pipeline**: Vector-based semantic search for context retrieval
- **Vector Storage**: FAISS for efficient embeddings
- **NLP Processing**: LangChain for text understanding
- **Embeddings**: user-selectable embedding model, routed through the same gateway

---

## 🎯 Getting Started

### Prerequisites

- Python 3.13
- An AI provider: either an OmniRoute gateway (Node >= 22, or Docker) **or** an OpenAI API key
- Node.js >= 22.22.2 — only if you run OmniRoute via npm
- 4GB+ RAM recommended
- macOS, Linux, or Windows

### 1. Installation & Setup

```bash
# Clone the repository
git clone https://github.com/jiten0709/TestRAGic.git
cd TestRAGic

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install

# Configure environment variables
cp src/.env.example src/.env
```

### 1b. Configure an AI provider

TestRAGic talks to exactly one OpenAI-compatible endpoint, chosen by `src/.env`.

**Option A — OmniRoute gateway (recommended).** OmniRoute is a local
OpenAI-compatible router: one endpoint, 350+ providers, built-in failover, and
per-provider credentials that live in its dashboard rather than in this repo.

```bash
npm i -g omniroute && omniroute          # boots the gateway + dashboard on :20128
# or, without Node:
docker run -d -p 127.0.0.1:20128:20128 -v omniroute-data:/app/data \
  diegosouzapw/omniroute:latest

# then in src/.env:
#   OMNIROUTE_BASE_URL=http://localhost:20128/v1
```

Open <http://localhost:20128> to connect providers. A fresh install answers with
no credentials at all, so `OMNIROUTE_LLM_MODEL=auto` works immediately.

**Option B — OpenAI directly.** Leave `OMNIROUTE_BASE_URL` empty and set
`OPENAI_API_KEY` in `src/.env`. The app behaves exactly as it did before the
migration.

#### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `OMNIROUTE_BASE_URL` | _(empty)_ | Gateway endpoint. **Setting this is what selects OmniRoute mode.** |
| `OMNIROUTE_API_KEY` | _(empty)_ | Gateway key. Optional — a local instance runs with `REQUIRE_API_KEY=false`. |
| `OMNIROUTE_LLM_MODEL` | `auto` | Persistent default chat model. `auto`, `auto/fast`, `provider/model`, … |
| `OMNIROUTE_EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Persistent default embedding model (1536 dims). |
| `OMNIROUTE_TIMEOUT` | `60` | Per-request timeout in seconds; bounds the fallback chain. |
| `TESTRAGIC_LLM_PROVIDER` | _(unset)_ | Force `omniroute` or `openai`, overriding the rule above. |
| `OPENAI_API_KEY` | _(empty)_ | Used when no gateway is configured. |
| `OPENAI_MODEL` | `gpt-4o-mini` | Default model in OpenAI-direct mode. |

`BASE_URL` is unrelated — it is the Playwright **test target** URL (`conftest.py`).

Model choice is also available in the UI (Settings → AI Provider, and the Test
Generation dropdown), but UI choices last for the session only, exactly like the
API key. Edit `src/.env` for a permanent default; Settings shows a copyable snippet.

### 2. Launch the Application

```bash
# Start Streamlit dashboard
streamlit run run_app.py

# Application will open at:
# Local:   http://localhost:8501
# Network: http://your-ip:8501
```

### 3. Verify Installation

```bash
# Test Playwright setup
pytest src/tests/generated/test_sample.py -v

# Verify all dependencies
python -c "import streamlit, playwright, openai, langchain; print('✅ All dependencies installed!')"

# Provider unit tests (no network, no browser)
pytest src/tests/unit -o addopts="" -q

# Live smoke test against a running OmniRoute gateway.
# Skipped automatically when no gateway is reachable.
OMNIROUTE_BASE_URL=http://localhost:20128/v1 \
  pytest src/tests/unit/test_live_omniroute.py -o addopts="" -q -s
```

### 4. First Test Run

1. **Configure the AI provider**
   - Go to Settings → AI Provider in the sidebar
   - Set the OmniRoute base URL, or an OpenAI API key
   - Pick the default LLM and embedding models
   - **Test connection** — it reports which provider actually answered

2. **Try Mock Data** (recommended first)
   - Use `"test"` as video URL
   - Generates sample test cases instantly
   - Verifies system works end-to-end

3. **Upload Your Video**
   - Local file: MP4, AVI, MOV, MKV, WebM
   - YouTube URL: Direct link or URL paste
   - Select test categories (Core Flows, Edge Cases, etc.)
   - Click "Process Video" → "Generate Test Cases"

---

## 💡 Use Cases

### For QA Teams

- Accelerate test case creation from requirements
- Reduce manual testing effort by 70%+
- Improve test coverage across browsers

### For Development Teams

- Quick regression testing after deployments
- Automated smoke tests in CI/CD pipelines
- Cross-browser compatibility validation

### For Product Teams

- Validate critical user journeys
- Ensure feature functionality across platforms
- Performance and accessibility compliance

### For Startups & SMEs

- Cost-effective QA automation
- Rapid testing setup for MVPs
- Scalable infrastructure without hiring

---

## ✅ Current Implementation Status

### Fully Implemented ✨

- Data Ingestion Agent (YouTube + transcript extraction)
- File Upload Support (MP4, AVI, MOV, MKV, WebM)
- AI-Powered Test Generation with RAG
- Streamlit Dashboard (multi-page interface)
- Playwright Test Execution Framework
- Vector Storage (FAISS)
- Mock Data Testing
- Multi-format Exports (JSON, Markdown)
- Multiple Input Methods (files, URLs, links)

---

## 🔮 Future Roadmap

### Q2 2026

- Parallel test execution across browsers
- Comprehensive historical reporting
- Advanced test analytics

### Q3 2026

- Claude & Gemini LLM integration
- CI/CD integration (GitHub Actions, Jenkins)
- Mobile app testing (React Native, Flutter)

### Q4 2026

- API testing (REST/GraphQL)
- Visual regression testing
- Security vulnerability assessment
- Cloud deployment (AWS, Azure, GCP)

---

## 🛠️ Troubleshooting

### ❌ AI provider issues

```bash
# "No AI provider configured"
#   Neither OMNIROUTE_BASE_URL nor OPENAI_API_KEY is set. Set one in src/.env,
#   or configure it on the Settings page.

# "Connection failed via omniroute at http://localhost:20128/v1"
#   The gateway is not running. Start it:
npm i -g omniroute && omniroute
curl -i http://localhost:20128/v1/models      # should return 200 + X-OmniRoute-* headers

# "Generated by: <model> (requested; gateway did not report)"
#   The response carried no X-OmniRoute-Provider/-Model header — usually a reverse
#   proxy stripping X-* headers. Attribution degrades honestly; it never guesses.

# "No model produced this result. Tried: ..."
#   Every rung of the fallback chain failed; the message names each one and why.
#   Any test cases shown are stubs, and are labelled as such.

# Model dropdown shows a static list with a warning
#   GET /v1/models was unreachable. The app never presents a stale list as live.
#   Press 🔄 next to the dropdown after fixing the gateway.

# NOTE: there is no API key *format* check any more. A gateway key is whatever
# its operator chose, and a local OmniRoute instance is keyless by default.
```

### ❌ YouTube Download Failures

```bash
# Error: Video download failed
# Solutions:
# 1. Try different YouTube URL
# 2. Use "test" for mock data instead
# 3. Check if video has download restrictions
# 4. Verify internet connection
```

### ❌ Playwright Browser Issues

```bash
# Error: Browser not found
# Solution:
playwright install
# Then restart the application
```

### ❌ Import Errors

```bash
# Error: Module not found
# Solutions:
# 1. Activate virtual environment
# 2. Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### ❌ Permission Errors

```bash
# Error: Permission denied when creating directories
# Solution:
chmod 755 src/data/
chmod 755 src/tests/
```

---

## 🤝 Support & Resources

| Resource          | Link                                                                     |
| ----------------- | ------------------------------------------------------------------------ |
| **Report Issues** | [GitHub Issues](https://github.com/jiten0709/TestRAGic/issues)           |
| **Documentation** | Check inline code comments and docstrings                                |
| **Testing**       | Use mock data ("test" URL) for safe testing                              |
| **Discussions**   | [GitHub Discussions](https://github.com/jiten0709/TestRAGic/discussions) |

---

## 📊 Performance Benchmarks

| Metric                   | Performance                 |
| ------------------------ | --------------------------- |
| Test Case Generation     | 2-5 min per video           |
| Parallel Execution Speed | 3-5x faster than sequential |
| Cross-Browser Coverage   | 3 browsers simultaneously   |
| Artifact Processing      | Real-time during execution  |
| Report Generation        | <30 seconds per test run    |

---

## 📝 License

This project is open source. See LICENSE file for details.

---

## 📞 Get in Touch

- 🐙 GitHub: [@jiten0709](https://github.com/jiten0709)
- 💼 LinkedIn: [Jiten Parmar](https://www.linkedin.com/in/jitenaparmar)

<h2 align="center">Built with ❤️ by <strong>Jiten Parmar</strong></h2>
