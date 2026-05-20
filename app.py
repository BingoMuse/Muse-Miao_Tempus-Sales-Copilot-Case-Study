"""
Tempus Sales Copilot — Multi-agent Streamlit prototype for Northwestern Medicine territory intelligence.
"""

from __future__ import annotations

import html
import io
import json
import logging
import os
import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from groq import Groq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = APP_DIR / "nm_oncologists_market_intelligence.csv"
DEFAULT_CRM = APP_DIR / "crm_notes.txt"
DEFAULT_KB_MD = APP_DIR / "tempus_product_knowledge_base.md"
DEFAULT_KB_PDF = APP_DIR / "tempus_product_knowledge_base.pdf"

MODEL_ID = "llama-3.3-70b-versatile"
PRIMARY_NAVY = "#0d47a1"
TEAL_ACCENT = "#00897b"
TOP3_HIGHLIGHT = "#e3f2fd"

WEIGHT_PATIENTS = 0.40
WEIGHT_PANEL_FIT = 0.35
WEIGHT_CRM = 0.25

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tempus_copilot")


# ---------------------------------------------------------------------------
# Theme & layout
# ---------------------------------------------------------------------------

def inject_theme() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: #ffffff;
        }}
        h1, h2, h3 {{
            color: {PRIMARY_NAVY};
        }}
        div[data-testid="stMetricValue"] {{
            color: {PRIMARY_NAVY};
        }}
        hr {{
            border-color: #e0e0e0;
        }}
        .rank-legend {{
            font-size: 0.85rem;
            color: #616161;
            padding: 0.75rem 1rem;
            background: #fafafa;
            border-left: 4px solid {PRIMARY_NAVY};
            border-radius: 4px;
            margin-bottom: 1rem;
        }}
        .card-navy {{
            background: #f5f8fc;
            border-left: 5px solid {PRIMARY_NAVY};
            padding: 1.25rem 1.5rem;
            border-radius: 8px;
            margin: 0.75rem 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }}
        .card-product {{
            background: #fff;
            border: 1px solid #e0e0e0;
            border-top: 4px solid {PRIMARY_NAVY};
            padding: 1.25rem 1.5rem;
            border-radius: 8px;
            margin: 0.75rem 0;
        }}
        .card-teal {{
            background: #f0faf9;
            border-left: 5px solid {TEAL_ACCENT};
            padding: 1.25rem 1.5rem;
            border-radius: 8px;
            margin: 0.75rem 0;
        }}
        .agent-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {PRIMARY_NAVY};
            font-weight: 600;
            margin-bottom: 0.35rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_card(title: str, body: str, css_class: str = "card-navy") -> None:
    safe_body = html.escape(body).replace("\n", "<br>")
    st.markdown(
        f'<div class="{css_class}">'
        f'<div class="agent-label">{title}</div>'
        f"<p style='margin:0;color:#212121;line-height:1.6;'>{safe_body}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# API & data loading
# ---------------------------------------------------------------------------

_groq_client: Groq | None = None


def get_api_key() -> str | None:
    try:
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return key
    except (FileNotFoundError, KeyError, AttributeError):
        pass
    return os.environ.get("GROQ_API_KEY")


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = get_api_key()
        if not api_key:
            st.error(
                "Groq API key not found. Set `GROQ_API_KEY` in "
                "`.streamlit/secrets.toml` or as an environment variable."
            )
            st.stop()
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def call_groq(
    system: str,
    user: str,
    max_tokens: int = 8192,
    temperature: float = 0.2,
) -> str:
    response = get_groq_client().chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content
    if not text:
        raise ValueError("Groq returned empty text.")
    return text.strip()


def parse_llm_json(raw: str) -> dict:
    json_match = re.search(r"\{[\s\S]*\}", raw)
    if not json_match:
        raise ValueError(f"Could not parse JSON from model: {raw[:500]}")
    return json.loads(json_match.group())


def normalize_name(name: str) -> str:
    n = name.lower().strip()
    n = re.sub(r",?\s*md\.?$", "", n, flags=re.I)
    n = re.sub(r"[^a-z0-9\s]", "", n)
    return re.sub(r"\s+", " ", n).strip()


def load_csv(source: bytes | Path) -> pd.DataFrame:
    if isinstance(source, Path):
        df = pd.read_csv(source)
    else:
        df = pd.read_csv(pd.io.common.BytesIO(source))
    required = {
        "Name",
        "Specialty",
        "Primary_Cancer_Focus",
        "Provider",
        "Seniority_Tier",
        "Review_Count",
        "Est_Monthly_Patients",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df


def load_crm_notes(source: bytes | str | Path) -> dict[str, str]:
    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8")
    elif isinstance(source, bytes):
        text = source.decode("utf-8")
    else:
        text = source

    notes: dict[str, str] = {}
    blocks = re.split(r"\n---+\n", text.strip())
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        match = re.search(r"Doctor:\s*(.+?)(?:\n|$)", block, re.I)
        note_match = re.search(r"Note:\s*(.+)", block, re.S | re.I)
        if match and note_match:
            doctor = match.group(1).strip()
            note = note_match.group(1).strip()
            notes[normalize_name(doctor)] = note
    return notes


def load_knowledge_base(md_path: Path | None, pdf_path: Path | None, upload_md: bytes | None) -> str:
    if upload_md:
        return upload_md.decode("utf-8")

    if md_path and md_path.exists():
        return md_path.read_text(encoding="utf-8")

    if pdf_path and pdf_path.exists():
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError("Install pypdf to read PDF knowledge base: pip install pypdf") from exc
        reader = PdfReader(str(pdf_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    raise FileNotFoundError(
        "Product knowledge base not found. Upload tempus_product_knowledge_base.md "
        "or place it (or the .pdf) in the app directory."
    )


def match_crm_to_providers(df: pd.DataFrame, crm_notes: dict[str, str]) -> dict[str, str | None]:
    """Map each provider name to CRM note text or None."""
    matched: dict[str, str | None] = {}
    crm_keys = list(crm_notes.keys())
    for _, row in df.iterrows():
        name = row["Name"]
        norm = normalize_name(name)
        note = crm_notes.get(norm)
        if note is None:
            for key, val in crm_notes.items():
                if key in norm or norm in key:
                    note = val
                    break
        if note is None:
            for key in crm_keys:
                last_csv = norm.split()[-1] if norm.split() else ""
                last_crm = key.split()[-1] if key.split() else ""
                if last_csv and last_csv == last_crm and last_csv != "md":
                    if norm.split()[0] == key.split()[0]:
                        note = crm_notes[key]
                        break
        matched[name] = note
    return matched


def min_max_normalize(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Ranking: panel fit via LLM
# ---------------------------------------------------------------------------

def score_panel_fit_batch(
    df: pd.DataFrame,
    kb_excerpt: str,
) -> dict[str, float]:
    """Return raw panel fit scores 1-5 per provider name."""
    providers_payload = []
    for _, row in df.iterrows():
        providers_payload.append(
            {
                "name": row["Name"],
                "specialty": row["Specialty"],
                "primary_cancer_focus": row["Primary_Cancer_Focus"],
            }
        )

    system = (
        "You are a Tempus oncology commercial intelligence analyst. Score each physician's "
        "alignment with Tempus core panels: xT CDx (solid tumors), xF/xF+ (liquid biopsy monitoring), "
        "xR (fusion-driven cancers: lung, leukemia, sarcoma), xT Heme (hematologic malignancies). "
        "Return ONLY valid JSON."
    )
    user = f"""Score each provider from 1 (poor fit) to 5 (excellent fit) based on how well their
Primary_Cancer_Focus aligns with Tempus panels.

Product context (abbreviated):
{kb_excerpt[:12000]}

Providers:
{json.dumps(providers_payload, indent=2)}

Respond with JSON only:
{{"scores": [{{"name": "<exact name>", "panel_fit_score": <1-5>}}, ...]}}
Include all {len(providers_payload)} providers. Use exact names from input."""

    raw = call_groq(system, user, max_tokens=4096)
    data = parse_llm_json(raw)
    scores: dict[str, float] = {}
    for item in data.get("scores", []):
        name = item.get("name", "")
        val = float(item.get("panel_fit_score", 3))
        scores[name] = max(1.0, min(5.0, val))

    for _, row in df.iterrows():
        if row["Name"] not in scores:
            scores[row["Name"]] = 3.0
    return scores


def compute_rankings(
    df: pd.DataFrame,
    crm_matched: dict[str, str | None],
    kb_text: str,
) -> pd.DataFrame:
    panel_raw = score_panel_fit_batch(df, kb_text)
    norm_patients = min_max_normalize(df["Est_Monthly_Patients"].astype(float))
    panel_series = df["Name"].map(panel_raw)
    norm_panel = (panel_series - 1) / 4.0

    crm_engagement = df["Name"].map(lambda n: 1.0 if crm_matched.get(n) else 0.0)

    composite = (
        WEIGHT_PATIENTS * norm_patients.values
        + WEIGHT_PANEL_FIT * norm_panel.values
        + WEIGHT_CRM * crm_engagement.values
    )

    ranked = df.copy()
    ranked["Panel_Fit_Score"] = panel_series.round(2)
    ranked["Norm_Patients"] = norm_patients.values
    ranked["Norm_Panel_Fit"] = norm_panel.values
    ranked["CRM_Engagement"] = crm_engagement.values
    ranked["Composite_Score"] = composite
    ranked["CRM_Status"] = ranked["Name"].map(
        lambda n: "Contacted" if crm_matched.get(n) else "Not Yet Contacted"
    )
    ranked = ranked.sort_values("Composite_Score", ascending=False).reset_index(drop=True)
    ranked.insert(0, "Rank", range(1, len(ranked) + 1))
    return ranked


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

def run_agent_1_market_intel(ranked_df: pd.DataFrame) -> str:
    logger.info("Agent 1 running... — Market Intelligence Specialist")
    rows = []
    for _, r in ranked_df.iterrows():
        rows.append(
            f"| {r['Rank']} | {r['Name']} | {r['Specialty']} | {r['Primary_Cancer_Focus']} | "
            f"{r['Provider']} | {r['Seniority_Tier']} | {r['Est_Monthly_Patients']} | "
            f"{r['Panel_Fit_Score']} | {r['CRM_Status']} | {r['Composite_Score']:.4f} |"
        )
    table = (
        "| Rank | Name | Specialty | Primary Cancer Focus | Institution | "
        "Seniority | Est. Monthly Patients | Panel Fit | CRM Status | Composite |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
        + "\n".join(rows)
    )
    logger.info("Agent 1 complete.")
    return table


def run_agent_2_crm_insights(
    ranked_df: pd.DataFrame,
    crm_matched: dict[str, str | None],
) -> str:
    logger.info("Agent 2 running... — CRM Insights Specialist")
    providers_payload = []
    for _, row in ranked_df.iterrows():
        name = row["Name"]
        note = crm_matched.get(name)
        providers_payload.append(
            {
                "name": name,
                "has_crm": bool(note),
                "crm_note": note or "",
            }
        )

    system = (
        "You are a CRM Insights Specialist. For each provider with has_crm true, extract "
        "their primary clinical objection or concern as ONE concise sentence. "
        "For providers with has_crm false, set summary to exactly: "
        "No prior interaction recorded. Return ONLY valid JSON."
    )
    user = f"""Providers:
{json.dumps(providers_payload, indent=2)}

Respond with JSON only:
{{"insights": [{{"name": "<exact name>", "summary": "<one sentence or no prior interaction>"}}, ...]}}
Include all {len(providers_payload)} providers. Use exact names from input."""

    raw = call_groq(system, user, max_tokens=4096)
    data = parse_llm_json(raw)
    summaries: dict[str, str] = {}
    for item in data.get("insights", []):
        summaries[item.get("name", "")] = item.get("summary", "No prior interaction recorded.")

    lines = []
    for _, row in ranked_df.iterrows():
        name = row["Name"]
        summary = summaries.get(name)
        if not summary:
            summary = (
                "No prior interaction recorded."
                if not crm_matched.get(name)
                else "CRM note on file; objection not summarized."
            )
        lines.append(f"**{name}**: {summary}")

    logger.info("Agent 2 complete.")
    return "\n\n".join(lines)


def run_agent_3_and_4_batch(
    ranked_df: pd.DataFrame,
    crm_matched: dict[str, str | None],
    agent2_output: str,
    kb_text: str,
    agent1_table: str,
) -> dict[str, dict[str, str]]:
    logger.info("Agent 3 & 4 Batched running... — Product Matcher + Manager Copilot")
    objection_summaries: dict[str, str] = {}
    for block in agent2_output.split("\n\n"):
        if block.startswith("**") and "**:" in block:
            name_part, _, summary = block.partition("**:")
            name = name_part.removeprefix("**").strip()
            objection_summaries[name] = summary.strip()

    providers_payload = []
    for _, row in ranked_df.iterrows():
        name = row["Name"]
        note = crm_matched.get(name)
        providers_payload.append(
            {
                "name": name,
                "rank": row["Rank"],
                "composite_score": round(row["Composite_Score"], 4),
                "drivers": f"Norm Patients: {row['Norm_Patients']:.3f}, Norm Panel Fit: {row['Norm_Panel_Fit']:.3f}, CRM Engagement: {row['CRM_Engagement']:.0f}",
                "specialty": row["Specialty"],
                "primary_cancer_focus": row["Primary_Cancer_Focus"],
                "est_monthly_patients": row["Est_Monthly_Patients"],
                "has_crm": bool(note),
                "crm_note": note or "",
                "crm_objection_summary": objection_summaries.get(name, ""),
            }
        )

    system = (
        "You are an expert Tempus oncology sales strategist with deep knowledge of genomic testing. "
        "Your job is to help sales reps walk into a 15-minute meeting with a specific oncologist and "
        "immediately answer: Why Tempus, Why Now, for this specific doctor? You reason by: "
        "(1) identifying the doctor's primary cancer type and patient population, "
        "(2) matching it to the Tempus panel with the strongest published clinical evidence for that cancer type, "
        "(3) connecting their specific CRM objection to a precise Tempus metric or feature that resolves it, and "
        "(4) synthesizing all of this into a confident, specific, human-sounding sales script. "
        "Always cite exact product names, turnaround times, and clinical statistics from the knowledge base. "
        "Never be generic. Return ONLY valid JSON."
    )
    user = f"""Knowledge base:
{kb_text[:20000]}

Agent 1 Context (Leaderboard Table):
{agent1_table[:3000]}

For EACH provider below, produce the following 5 fields:

- part_a — Proactive product recommendation: First identify the doctor's primary cancer type from their cancer focus. Then identify the 1-2 Tempus panels most relevant to that cancer type from the knowledge base. Cite the specific clinical evidence statistic that makes this panel valuable for their patient population (e.g. therapy match rates, fusion detection improvements, turnaround time). 3-5 sentences.

- part_b — Objection-specific product match: If has_crm is true, identify the SINGLE most relevant Tempus capability, metric, or feature from the knowledge base that directly addresses their CRM objection (crm_note / crm_objection_summary). Reference exact product names, turnaround times, or clinical statistics. 2-4 sentences. If has_crm is false, set to an empty string.

- snapshot — Provider Snapshot: Exactly 2 sentences explaining why this doctor is ranked where they are (rank and composite_score), referencing their composite score drivers (patient volume, panel fit, CRM engagement).

- objection_handler — If CRM history exists: Write a 3-5 sentence response that a sales rep would say out loud directly to this doctor. Start by acknowledging their specific concern by name, then immediately pivot to the precise Tempus metric or feature that addresses it, citing exact numbers from the knowledge base. End with a concrete next step. If no CRM history: Write a compelling cold outreach opening that references their specific cancer focus and patient volume.

- elevator_pitch — Write exactly what a sales rep would say in the first 30 seconds of a meeting with this specific doctor. Address them by name. Reference their cancer focus and why Tempus is relevant right now for their patients. Weave in the product recommendation and one key clinical statistic. End with a question or hook that invites engagement. Maximum 5 sentences. Sound like a confident human, not a brochure.

Providers:
{json.dumps(providers_payload, indent=2)}

Respond with JSON only:
{{
  "providers": [
    {{
      "name": "<exact name>",
      "part_a": "...",
      "part_b": "...",
      "snapshot": "...",
      "objection_handler": "...",
      "elevator_pitch": "..."
    }},
    ...
  ]
}}
Include all {len(providers_payload)} providers. Use exact names from input."""

    raw = call_groq(system, user, max_tokens=8192)
    data = parse_llm_json(raw)
    results: dict[str, dict[str, str]] = {}
    for item in data.get("providers", []):
        name = item.get("name", "")
        if name:
            results[name] = {
                "part_a": item.get("part_a", ""),
                "part_b": item.get("part_b", ""),
                "snapshot": item.get("snapshot", ""),
                "objection_handler": item.get("objection_handler", ""),
                "elevator_pitch": item.get("elevator_pitch", ""),
            }

    for _, row in ranked_df.iterrows():
        name = row["Name"]
        if name not in results:
            results[name] = {
                "part_a": "", "part_b": "", "snapshot": "",
                "objection_handler": "", "elevator_pitch": ""
            }

    logger.info("Agent 3 & 4 Batched complete — %d providers processed in one call.", len(results))
    return results


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    defaults = {
        "pipeline_complete": False,
        "ranked_df": None,
        "crm_matched": None,
        "kb_text": None,
        "agent1_output": None,
        "agent2_output": None,
        "agent3_4_output": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def run_pipeline(
    df: pd.DataFrame,
    crm_matched: dict[str, str | None],
    kb_text: str,
) -> None:
    get_groq_client()
    with st.spinner("Analyzing clinical workflows..."):
        status = st.empty()
        status.info("Computing composite rankings and panel fit scores…")
        ranked_df = compute_rankings(df, crm_matched, kb_text)
        st.session_state.ranked_df = ranked_df

        status.info("Agent 1 running — Market Intelligence Specialist…")
        st.session_state.agent1_output = run_agent_1_market_intel(ranked_df)

        status.info("Agent 2 running — CRM Insights Specialist…")
        agent2_output = run_agent_2_crm_insights(ranked_df, crm_matched)
        st.session_state.agent2_output = agent2_output

        status.info("Agent 3 & 4 running — Product Matcher & Manager Copilot (batching 12 providers)…")
        st.session_state.agent3_4_output = run_agent_3_and_4_batch(
            ranked_df,
            crm_matched,
            agent2_output,
            kb_text,
            st.session_state.agent1_output,
        )

        st.session_state.pipeline_complete = True
        status.success("Pipeline complete. Select a physician in Sales Execution Hub.")


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def style_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    display = df[
        [
            "Rank",
            "Name",
            "Specialty",
            "Primary_Cancer_Focus",
            "Provider",
            "Seniority_Tier",
            "Est_Monthly_Patients",
            "Panel_Fit_Score",
            "Composite_Score",
            "CRM_Status",
        ]
    ].copy()
    display = display.rename(
        columns={
            "Primary_Cancer_Focus": "Primary Cancer Focus",
            "Provider": "Institution",
            "Seniority_Tier": "Seniority Tier",
            "Est_Monthly_Patients": "Est. Monthly Patients",
            "Panel_Fit_Score": "Panel Fit Score",
            "Composite_Score": "Composite Score",
            "CRM_Status": "CRM Status",
        }
    )
    display["Composite Score"] = display["Composite Score"].map(lambda x: f"{x:.4f}")
    return display


def render_leaderboard_tab() -> None:
    st.subheader("Territory Leaderboard")
    st.markdown(
        '<div class="rank-legend">'
        "<strong>Composite ranking formula:</strong> "
        f"{WEIGHT_PATIENTS:.0%} normalized monthly patient volume + "
        f"{WEIGHT_PANEL_FIT:.0%} normalized Tempus panel fit (LLM 1–5) + "
        f"{WEIGHT_CRM:.0%} CRM engagement (1.0 if contacted, 0.0 if not)."
        "</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.ranked_df is None:
        st.info("Click **Run Pipeline** in the sidebar to generate rankings and agent outputs.")
        return

    display_df = style_leaderboard(st.session_state.ranked_df)

    def highlight_top3(row: pd.Series) -> list[str]:
        if row.name < 3:
            return [f"background-color: {TOP3_HIGHLIGHT}"] * len(row)
        return [""] * len(row)

    styled = display_df.style.apply(highlight_top3, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption("Top 3 providers highlighted in light blue.")


def render_sales_hub_tab() -> None:
    st.subheader("Sales Execution Hub")

    if not st.session_state.pipeline_complete:
        st.info("Run the pipeline first to unlock physician-specific copilot outputs.")
        return

    ranked = st.session_state.ranked_df
    names = ranked["Name"].tolist()
    default_idx = 0

    selected = st.selectbox(
        "Select physician (ranked by composite score)",
        options=names,
        index=default_idx,
        format_func=lambda n: f"#{ranked.loc[ranked['Name']==n, 'Rank'].iloc[0]} — {n}",
    )

    if not selected:
        return

    brief = st.session_state.agent3_4_output.get(selected, {})

    st.markdown("---")
    st.markdown("#### Provider Snapshot")
    render_card("Agent 4 — Ranking Rationale", brief.get("snapshot", ""), "card-navy")

    st.markdown("#### Recommended Tempus Products")
    product_text = brief.get("part_a", "No recommendation available.")
    render_card("Agent 3 — Proactive Product Match", product_text, "card-product")

    st.markdown("#### The Objection Handler")
    render_card("Agent 4 — Objection Handler", brief.get("objection_handler", ""), "card-navy")

    st.markdown("#### The 30-Second Elevator Pitch")
    render_card("Agent 4 — Elevator Pitch", brief.get("elevator_pitch", ""), "card-teal")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Tempus Sales Copilot",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()
    init_session_state()

    st.title("Tempus Sales Copilot")
    st.caption("Northwestern Medicine territory intelligence · Multi-agent clinical workflow synthesis")

    with st.sidebar:
        st.header("Data sources")
        st.markdown(
            f"<span style='color:{PRIMARY_NAVY};font-weight:600;'>File ingestion</span>",
            unsafe_allow_html=True,
        )

        csv_upload = st.file_uploader("Market intelligence CSV", type=["csv"])
        crm_upload = st.file_uploader("CRM notes (txt)", type=["txt"])
        kb_upload = st.file_uploader("Product knowledge base (md or pdf)", type=["md", "pdf"])

        st.divider()
        run_clicked = st.button("Run Pipeline", type="primary", use_container_width=True)

        if st.session_state.pipeline_complete:
            st.success("Pipeline ready")
            st.caption("Agent 3 & 4 data pre-computed for all physicians")

    try:
        csv_bytes = csv_upload.read() if csv_upload else None
        df = load_csv(csv_bytes if csv_bytes else DEFAULT_CSV)

        crm_source: bytes | str | Path
        if crm_upload:
            crm_source = crm_upload.read()
        else:
            crm_source = DEFAULT_CRM
        crm_notes = load_crm_notes(crm_source)
        crm_matched = match_crm_to_providers(df, crm_notes)
        st.session_state.crm_matched = crm_matched

        kb_text: str
        if kb_upload:
            raw_kb = kb_upload.read()
            if kb_upload.name.lower().endswith(".pdf"):
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(raw_kb))
                kb_text = "\n".join(page.extract_text() or "" for page in reader.pages)
            else:
                kb_text = raw_kb.decode("utf-8")
        else:
            kb_text = load_knowledge_base(
                DEFAULT_KB_MD if DEFAULT_KB_MD.exists() else None,
                DEFAULT_KB_PDF if DEFAULT_KB_PDF.exists() else None,
                None,
            )
        st.session_state.kb_text = kb_text

    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

    if run_clicked:
        try:
            run_pipeline(df, crm_matched, kb_text)
            st.rerun()
        except Exception as exc:
            logger.exception("Pipeline failed")
            st.error(f"Pipeline error: {exc}")

    tab1, tab2 = st.tabs(["Territory Leaderboard", "Sales Execution Hub"])
    with tab1:
        render_leaderboard_tab()
    with tab2:
        render_sales_hub_tab()


if __name__ == "__main__":
    main()
