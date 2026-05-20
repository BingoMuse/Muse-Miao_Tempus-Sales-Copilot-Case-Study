# Tempus Sales Copilot: A Multi-Agent AI Solution for Oncology Territory Intelligence

An intelligent, multi-agent sales enablement prototype designed to bridge the data-to-insight gap for oncology sales representatives. Before high-stakes clinical meetings, sales teams often spend hours synthesizing disparate datasets. This solution automates territory intelligence by cross-referencing market intelligence, prior CRM histories, and an official product knowledge base to output a prioritized provider leaderboard, structured objection handlers, and highly personalized 30-second elevator pitches.

## 📊 Data Sources & Curation

The platform ingests and fuses three distinct data streams to simulate realistic territory conditions:
* **Market Intelligence:** Sourced directly from the Northwestern Medicine physician portal, covering 12 oncologists. The data spans across 4 high-alignment subspecialties matching Tempus's clinical portfolio (*xT, xF, xR, xT Heme*), exclusively targeting MDs with independent test ordering authority across main campuses and suburban affiliates.
* **Product Knowledge Base:** Comprehensive product specifications extracted directly from 10 scraped pages of Tempus.com. This includes data on *xT CDx, xF/xF+ liquid biopsy, xR, xT Heme, clinical trial matching, and EHR integrations*. A strict zero-hallucination guardrail is implemented, bounding agent outputs strictly to verified web statistics.
* **CRM Notes:** Digitally synthesized interaction records covering authentic oncology sales objections (e.g., reimbursement, turnaround time, EHR integrations). To mimic real-world data environments, an intentional whitespace of 4 out of 12 physicians contains no prior CRM history to simulate cold leads.

## 🚀 Accessing the Prototypes

### 🌐 Standalone Pre-Computed Prototype
Due to external API rate limits on free-tier LLM endpoints, a reliable, fully rendered HTML prototype containing pre-computed agent outputs generated via unified JSON batching is available for immediate viewing:
* **Live Interactive Link:** [View the Live Sales Copilot Prototype](https://github.com/BingoMuse/Muse-Miao_Tempus-Sales-Copilot-Case-Study/blob/3c0e16ce30daa893a01725c487ae2a4ce4c061c9/prototype/precomputed_copilot_demo.html)

### 💻 Running the Live Streamlit Pipeline App
To launch the interactive multi-agent application locally on your machine, navigate to your root project directory and execute the following three commands in your terminal:

```bash
# 1. Activate your Python virtual environment
source venv/bin/activate

# 2. Install the required pipeline packages
pip install -r prototype/pipeline-app/requirements.txt

# 3. Spin up the local Streamlit dashboard
streamlit run prototype/pipeline-app/app.py
```

> ⚠️ **Note on Live Execution & API Limits:** Due to strict daily token and rate limits on free-tier LLM endpoints, executing the live pipeline will successfully run **Agent 1 (Market Intelligence)** and **Agent 2 (CRM Insights)** to dynamically compute composite impact scores and generate the live ranked territory leaderboard. However, downstream execution will hit the free-tier ceiling when attempting the 25+ individual LLM calls required for the remaining agents. To evaluate the complete end-to-end multi-agent output without endpoint restrictions, please open the standalone pre-computed prototype (`prototype/precomputed_demo.html`).

## 📂 Project Deliverables

* **Slide Deck:** View the full strategic product presentation and problem decomposition at [slides/tempus_sales_copilot_deck.pdf](slides/tempus_sales_copilot_deck.pdf).
* **Demo Video:** Watch a walkthrough of the dual-tab operational dashboard interface at [assets/demo_video.mp4](assets/demo_video.mp4).
