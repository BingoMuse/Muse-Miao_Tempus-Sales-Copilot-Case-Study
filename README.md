# Tempus Sales Copilot: A Multi-Agent AI Solution for Oncology Territory Intelligence

An intelligent, multi-agent sales enablement prototype designed to bridge the data-to-insight gap for oncology sales representatives. Before high-stakes clinical meetings, sales teams often spend hours synthesizing disparate datasets. This solution automates territory intelligence by cross-referencing market intelligence, prior CRM histories, and an official product knowledge base to output a prioritized provider leaderboard, structured objection handlers, and highly personalized 30-second elevator pitches.

---

## 🏗️ Repository Structure

```
tempus-sales-copilot/
├── data/                              # Market intelligence CSV, CRM notes, knowledge base
├── prototype/
│   ├── pipeline-app/                  # Streamlit multi-agent app (app.py + requirements.txt)
│   └── precomputed_copilot_demo.html  # Standalone pre-computed demo
├── slides/
│   └── tempus_sales_copilot_deck.pdf
└── assets/
    └── demo_video.mp4
```

---

## 🤖 Multi-Agent Pipeline

The pipeline runs 4 specialized agents with **3 total LLM calls** — reduced from 25+ via unified JSON batching:

| Agent | Role | LLM |
|---|---|---|
| **Agent 1 — Market Intelligence Specialist** | Ingests CSV + composite scores → outputs ranked leaderboard | None (deterministic Python) |
| **Agent 2 — CRM Insights Specialist** | Extracts one-sentence objection per doctor from CRM notes | 1 batched call (all 12 doctors) |
| **Agent 3 — Product Matcher** | Maps cancer focus → most relevant Tempus panel(s) with clinical evidence; maps CRM objection → specific Tempus metric that resolves it | Merged with Agent 4 |
| **Agent 4 — Manager Copilot** | Synthesizes Provider Snapshot, Objection Handler, and 30-Second Elevator Pitch per doctor | 1 batched call (all 12 doctors) |

Agents 1 and 2 run in parallel via Python threading. Agents 3 and 4 are merged into a single batched LLM call for rate limit optimization.

---

## 📊 Data Sources & Curation

The platform ingests and fuses three distinct data streams to simulate realistic territory conditions:

- **Market Intelligence:** Sourced directly from the Northwestern Medicine physician portal, covering 12 oncologists across 4 high-alignment subspecialties matching Tempus's clinical portfolio (xT, xF, xR, xT Heme), exclusively targeting MDs with independent test ordering authority across main campuses and suburban affiliates.

- **Product Knowledge Base:** Comprehensive product specifications extracted from 10 scraped pages of Tempus.com — including xT CDx, xF/xF+ liquid biopsy, xR, xT Heme, clinical trial matching, and EHR integrations. A strict zero-hallucination guardrail bounds all agent outputs to verified web statistics only.

- **CRM Notes:** Synthesized interaction records covering authentic oncology sales objections (reimbursement, turnaround time, EHR integration, competing vendors, tissue sufficiency, data privacy). 4 of 12 physicians have no prior CRM history, replicating real-world cold lead conditions.

---

## 🚀 Accessing the Prototypes

### 🌐 Standalone Pre-Computed Prototype (Recommended)

A fully rendered, interactive HTML prototype with pre-computed agent outputs is available for immediate viewing — no installation or API key required:

**[View the Live Sales Copilot Prototype](https://bingomuse.github.io/Muse-Miao_Tempus-Sales-Copilot-Case-Study/prototype/precomputed_copilot_demo.html)**

Or open the file directly in any browser:

```
prototype/precomputed_copilot_demo.html
```

> **Why pre-computed?** The prototype was built under free-tier API constraints. Although the pipeline was refactored to use only 3 batched LLM calls (down from 25+), daily token limits across Anthropic, Groq, Mistral, and Google Gemini were exhausted during development and debugging. All Agent 3+4 outputs were therefore generated in advance using Claude and the scraped Tempus knowledge base, then hardcoded into the HTML prototype — guaranteeing zero runtime dependency and hallucination-free outputs grounded in real Tempus product data.

---

### 💻 Running the Live Streamlit Pipeline App

**Prerequisites:** Python 3.9+, free Groq API key ([console.groq.com](https://console.groq.com))

```bash
# 1. Activate your Python virtual environment
source venv/bin/activate

# 2. Install the required pipeline packages
pip install -r prototype/pipeline-app/requirements.txt

# 3. Add your Groq API key
echo 'GROQ_API_KEY = "your_key_here"' > prototype/pipeline-app/.streamlit/secrets.toml

# 4. Spin up the local Streamlit dashboard
streamlit run prototype/pipeline-app/app.py
```

> **Note on live execution:** The live pipeline will successfully run Agent 1 (deterministic ranking) and Agent 2 (CRM insights) immediately. Agent 3+4 requires a Groq API key with sufficient daily token quota. If the free-tier daily limit is reached, open the standalone HTML prototype for the full end-to-end experience.

---

## 📂 Project Deliverables

- **Slide Deck:** [slides/tempus_sales_copilot_deck.pdf](slides/tempus_sales_copilot_deck.pdf)
- **Demo Video:** [assets/demo_video.mp4](assets/demo_video.mp4)

---

## 🛠️ Tools Used

Built with **Cursor AI** · **Streamlit** · **Groq API (LLaMA 3.3-70b-versatile)** · **Python** · **Claude** (knowledge base scraping, data curation, pre-computed outputs)
