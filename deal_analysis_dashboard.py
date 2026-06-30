import io
import re

import numpy as np
import pandas as pd
import streamlit as st

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from pptx import Presentation
except Exception:
    Presentation = None

try:
    from docx import Document
except Exception:
    Document = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="LSH Deal Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)


# ======================================================
# SYNTHETIC DATA
# ======================================================

SYNTHETIC_RFP = """
Synthetic LSH RFP: Global Application Services and Modernization Partner

Client: Global biopharma company operating across commercial, clinical, regulatory, manufacturing, and quality functions.

Scope:
The client seeks a strategic partner for Application Services Management and Application Development across SAP S/4HANA, Veeva Vault, Salesforce Health Cloud, Workday, ServiceNow, and cloud-native applications. The partner will provide L1, L2 and L3 support, enhancement delivery, release management, integration support, DevSecOps, and transition services.

Mandatory Requirements:
- Provide a clear transition plan with milestones, risks, dependencies, governance and RACI.
- Support GxP validated systems with CSV/CSA practices and 21 CFR Part 11 awareness.
- Demonstrate life sciences domain experience in regulatory, quality, pharmacovigilance, clinical and manufacturing.
- Provide SLA management for incident, problem, change, release and service request processes.
- Propose an automation-led operating model with shift-left, self-healing, AIOps, and ticket deflection.
- Include pricing model, productivity commitments, assumptions, risk sharing and commercial levers.
- Demonstrate delivery confidence with case studies, accelerators, staffing plan, global delivery model and PMO.
- Identify win themes and explain why the bidder is differentiated versus large SI competitors.

Business Priorities:
The client wants reduced run cost, faster release cycles, lower operational risk, stronger business alignment, improved user experience, and practical AI/GenAI use cases that improve productivity without compromising compliance.

Evaluation Criteria:
RFP compliance, LSH domain fit, commercial competitiveness, AI and automation innovation, transition confidence, technology fit, and differentiated value proposition.
"""

SYNTHETIC_SOLUTION = """
Synthetic Solution Response: LSH Application Services Transformation

Executive Summary:
We propose a life sciences aligned application services model for SAP S/4HANA, Veeva Vault, Salesforce, Workday, ServiceNow and cloud-native applications. Our solution combines ASM, AD, DevSecOps, automation, AI/GenAI and domain-led governance to reduce run cost and improve release velocity.

Win Themes:
1. LSH-first operating model with GxP, CSV/CSA and regulatory awareness embedded into delivery.
2. Automation and AI-led productivity with AIOps, ticket deflection, self-healing and knowledge assistants.
3. Commercial flexibility with productivity commitments, transparent assumptions and risk-sharing options.
4. Proven transition playbook with governance, PMO, RACI, milestones and risk mitigation.
5. Platform depth across SAP, Veeva, Salesforce, Workday, ServiceNow and cloud integration.

Scope and Service Model:
The service model covers L1, L2 and L3 support, incident, problem, change, release, service request, enhancement and minor project delivery. We will use ITIL, Agile and DevSecOps practices with global delivery across onsite, nearshore and offshore teams. Shift-left and hyperautomation will reduce avoidable tickets.

LSH Domain:
The response includes support for regulatory, quality, clinical, commercial and manufacturing processes. Validated systems will follow GxP, CSV/CSA and 21 CFR Part 11 expectations, with audit-ready documentation and compliance controls.

Technology:
The architecture includes SAP S/4HANA, Veeva Vault, Salesforce Health Cloud, Workday, ServiceNow, API integration, CI/CD, observability, Azure, AWS and enterprise architecture governance.

AI and GenAI:
We propose GenAI knowledge management, ticket summarization, automated resolution recommendations, AIOps event correlation, predictive incident prevention, and responsible AI governance. These use cases are designed to improve productivity, reduce mean time to resolve and improve user experience.

Commercials:
The proposal includes a blended global rate card, baseline TCO, year-on-year productivity commitments, outcome-based gainshare options, assumptions, dependencies, SLA credits and risk sharing. Cost takeout will be tracked through automation benefits and demand reduction.

Delivery Confidence:
We will use a structured transition plan with discovery, knowledge transfer, shadow support, reverse shadow, stabilization and steady state. Governance includes weekly operations review, monthly service review and quarterly business review. The team includes PMO, solution architects, platform leads and LSH SMEs.

Modernization:
We will identify technical debt, rationalization opportunities, API-led integration patterns, cloud migration candidates and business outcome KPIs for modernization waves.

Data & Analytics:
We will provide KPI dashboards, SLA reporting, BI integration and data quality governance where required.
"""


# ======================================================
# SCORING CONFIG
# ======================================================

SCORING_DIMENSIONS = [
    {
        "id": "rfp_compliance",
        "name": "RFP Compliance & Responsiveness",
        "weight": 15,
        "description": "Checks whether the response addresses explicit RFP scope, deliverables, timelines, SLAs, assumptions, and mandatory asks.",
        "keywords": [
            "scope", "requirement", "deliverable", "compliance", "SLA",
            "service level", "transition", "timeline", "dependency",
            "assumption", "governance", "acceptance criteria", "RACI"
        ],
    },
    {
        "id": "lsh_domain_alignment",
        "name": "LSH Domain Alignment",
        "weight": 10,
        "description": "Rates life sciences relevance including GxP, validation, pharmacovigilance, clinical, manufacturing, quality, commercial, regulatory and patient context.",
        "keywords": [
            "life sciences", "pharma", "biotech", "medtech", "GxP",
            "GMP", "GCP", "GLP", "CSV", "CSA", "validation",
            "21 CFR Part 11", "HIPAA", "pharmacovigilance", "clinical",
            "regulatory", "quality", "patient", "manufacturing", "commercial"
        ],
    },
    {
        "id": "ai_readiness",
        "name": "AI & GenAI Differentiation",
        "weight": 10,
        "description": "Rates AI/GenAI relevance, practical use cases, productivity impact, intelligent automation, responsible AI, and measurable value.",
        "keywords": [
            "AI", "GenAI", "generative AI", "agentic AI", "machine learning",
            "intelligent automation", "automation intelligence", "copilot",
            "predictive", "responsible AI", "AI governance", "model governance",
            "LLM", "AIOps", "self-healing", "ticket deflection",
            "knowledge management", "productivity"
        ],
    },
    {
        "id": "commercial_solution",
        "name": "Commercial & Contracting Strength",
        "weight": 10,
        "description": "Measures pricing clarity, productivity commitments, outcome-based elements, assumptions, dependencies, risk sharing, and commercial competitiveness.",
        "keywords": [
            "pricing", "commercial", "rate card", "TCO", "productivity",
            "gainshare", "outcome-based", "fixed price", "T&M",
            "assumptions", "dependencies", "credits", "penalties",
            "SLA credits", "risk sharing", "cost takeout", "savings",
            "benchmark"
        ],
    },
    {
        "id": "win_themes",
        "name": "Win Themes & Differentiation",
        "weight": 10,
        "description": "Rates client-specific win themes, quantified differentiation, executive messaging, competitor-aware positioning, and compelling reasons to select the bidder.",
        "keywords": [
            "win theme", "differentiator", "client-specific", "why us",
            "value proposition", "innovation", "co-create",
            "transformation partner", "domain expertise",
            "competitive advantage", "executive message", "strategic alignment",
            "business outcome", "proof point", "accelerator", "unique"
        ],
    },
    {
        "id": "technology_fit",
        "name": "Technology & Platform Fit",
        "weight": 9,
        "description": "Rates platform and technical depth across SAP, Salesforce, Veeva, Workday, ServiceNow, cloud, integration, automation, and enterprise platforms.",
        "keywords": [
            "SAP", "S/4HANA", "Salesforce", "Veeva", "Vault",
            "Workday", "ServiceNow", "AWS", "Azure", "GCP",
            "integration", "API", "automation", "DevOps", "CI/CD",
            "observability", "enterprise architecture"
        ],
    },
    {
        "id": "service_model",
        "name": "Service Model & Operating Model",
        "weight": 9,
        "description": "Assesses managed services, ITIL, Agile, DevSecOps, global delivery, shift-left, automation-led support, governance cadence, and transition approach.",
        "keywords": [
            "managed services", "ASM", "AD", "Agile", "DevSecOps",
            "ITIL", "global delivery", "onsite", "offshore", "nearshore",
            "shift-left", "hyperautomation", "L1", "L2", "L3",
            "incident", "problem", "change", "release", "transition"
        ],
    },
    {
        "id": "delivery_confidence",
        "name": "Delivery Confidence & Proof Points",
        "weight": 9,
        "description": "Checks similar case studies, accelerators, reusable assets, staffing plan, ramp-up, governance, measurable proof, and delivery credibility.",
        "keywords": [
            "case study", "proof point", "accelerator", "reusable asset",
            "reference", "experience", "staffing", "ramp-up",
            "delivery plan", "milestone", "governance", "program management",
            "PMO", "risk mitigation", "lessons learned"
        ],
    },
    {
        "id": "modernization_value",
        "name": "Modernization & Business Value",
        "weight": 7,
        "description": "Measures modernization, transformation, measurable outcomes, technical debt reduction, cost improvement, speed, and quality impact.",
        "keywords": [
            "modernization", "transformation", "rationalization",
            "technical debt", "cloud migration", "API-led", "microservices",
            "value realization", "business outcome", "cost reduction",
            "faster release", "ROI", "KPI", "benefit case"
        ],
    },
    {
        "id": "compliance_security_risk",
        "name": "Compliance, Security & Risk",
        "weight": 7,
        "description": "Assesses cybersecurity, privacy, regulated systems, validation, auditability, risk management, and compliance controls.",
        "keywords": [
            "security", "privacy", "cyber", "ISO 27001", "SOC 2",
            "GDPR", "HIPAA", "Part 11", "audit", "validation",
            "risk", "controls", "data protection", "vulnerability",
            "disaster recovery", "business continuity"
        ],
    },
    {
        "id": "data_analytics",
        "name": "Data & Analytics",
        "weight": 4,
        "description": "Rates data foundation, reporting, analytics, governance, dashboards, BI, and metrics relevance.",
        "keywords": [
            "data governance", "MDM", "data quality", "data lake",
            "lakehouse", "analytics", "dashboard", "Power BI",
            "Tableau", "metadata", "lineage", "reporting", "BI",
            "KPI dashboard"
        ],
    },
]


# ======================================================
# TEXT EXTRACTION
# ======================================================

def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def extract_pdf(file_obj):
    if PdfReader is None:
        raise RuntimeError("pypdf is not installed. Add pypdf to requirements.txt.")
    reader = PdfReader(file_obj)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_pptx(file_obj):
    if Presentation is None:
        raise RuntimeError("python-pptx is not installed. Add python-pptx to requirements.txt.")
    prs = Presentation(file_obj)
    chunks = []
    for i, slide in enumerate(prs.slides, start=1):
        chunks.append(f"\n--- Slide {i} ---")
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                chunks.append(shape.text)
            if hasattr(shape, "has_table") and shape.has_table:
                for row in shape.table.rows:
                    chunks.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(chunks)


def extract_docx(file_obj):
    if Document is None:
        raise RuntimeError("python-docx is not installed. Add python-docx to requirements.txt.")
    doc = Document(file_obj)
    chunks = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            chunks.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(chunks)


def extract_text(uploaded_file):
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    buffer = io.BytesIO(data)

    if name.endswith(".pdf"):
        return extract_pdf(buffer)

    if name.endswith(".pptx"):
        return extract_pptx(buffer)

    if name.endswith(".docx"):
        return extract_docx(buffer)

    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")

    raise ValueError("Unsupported file type. Upload PDF, PPTX, DOCX, or TXT.")


# ======================================================
# SCORING ENGINE
# ======================================================

def keyword_hits(text, keywords):
    text_lower = text.lower()
    found = []

    for keyword in keywords:
        if keyword.lower() in text_lower:
            found.append(keyword)

    return len(found), found


def cosine_score(query_terms, response_text):
    query = " ".join(query_terms)
    docs = [query, response_text]

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=10000
        )
        matrix = vectorizer.fit_transform(docs)
        score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return float(score)
    except Exception:
        return 0.0


def score_dimension(rfp_text, solution_text, dimension):
    keywords = dimension.get("keywords", [])

    rfp_hit_count, rfp_hits = keyword_hits(rfp_text, keywords)
    solution_hit_count, solution_hits = keyword_hits(solution_text, keywords)

    semantic_similarity = cosine_score(keywords + rfp_hits, solution_text)

    rfp_relevance = min(1.0, rfp_hit_count / max(3, min(8, len(keywords))))
    solution_coverage = min(1.0, solution_hit_count / max(4, min(10, len(keywords))))
    semantic_coverage = min(1.0, semantic_similarity * 6)

    if rfp_hit_count == 0:
        raw_score = (0.35 * solution_coverage + 0.25 * semantic_coverage) * 5
    else:
        raw_score = (
            0.25 * rfp_relevance
            + 0.50 * solution_coverage
            + 0.25 * semantic_coverage
        ) * 5

    score_0_to_5 = round(max(0.0, min(5.0, raw_score)), 2)
    weighted_score = round((score_0_to_5 / 5.0) * float(dimension["weight"]), 2)

    missing_keywords = [kw for kw in keywords if kw not in solution_hits]

    if score_0_to_5 < 2.5:
        feedback = (
            "Major gap. Add explicit treatment, proof points, measurable commitments, "
            "and client-specific linkage."
        )
    elif score_0_to_5 < 3.75:
        feedback = (
            "Partial coverage. Strengthen specificity, evidence, quantified benefits, "
            "and solution-to-RFP linkage."
        )
    else:
        feedback = (
            "Reasonably covered. Improve further with page-level evidence, quantified benefits, "
            "and stronger differentiators."
        )

    if missing_keywords:
        feedback += " Consider adding: " + ", ".join(missing_keywords[:8]) + "."

    return {
        "Dimension": dimension["name"],
        "Weight": dimension["weight"],
        "Score / 5": score_0_to_5,
        "Weighted Score": weighted_score,
        "RFP Signals": ", ".join(rfp_hits[:12]) if rfp_hits else "None detected",
        "Solution Signals": ", ".join(solution_hits[:12]) if solution_hits else "None detected",
        "Feedback": feedback,
    }


def score_deal(rfp_text, solution_text):
    rows = []

    for dimension in SCORING_DIMENSIONS:
        rows.append(score_dimension(rfp_text, solution_text, dimension))

    scorecard = pd.DataFrame(rows)
    total_score = round(float(scorecard["Weighted Score"].sum()), 2)

    if total_score >= 80:
        rating = "Green"
        verdict = "Strong candidate for submission, subject to final SME, legal, and commercial review."
    elif total_score >= 65:
        rating = "Amber"
        verdict = "Potentially submit-ready, but targeted improvements are recommended before submission."
    else:
        rating = "Red"
        verdict = "Not submit-ready. Address major gaps before governance approval."

    top_gaps = (
        scorecard
        .sort_values(["Score / 5", "Weight"], ascending=[True, False])
        .head(5)[["Dimension", "Score / 5", "Feedback"]]
    )

    return scorecard, total_score, rating, verdict, top_gaps


# ======================================================
# STREAMLIT UI
# ======================================================

st.title("📊 LSH Deal Analysis Dashboard")
st.caption(
    "RFP response readiness engine for ASM, AD, Modernization, AI, Data & Analytics, "
    "SAP, Salesforce, Veeva, Workday and related LSH pursuits."
)

with st.sidebar:
    st.header("Deal Scoring Model")

    total_weight = sum(d["weight"] for d in SCORING_DIMENSIONS)
    st.write(f"Total weight: **{total_weight}**")

    weights_df = pd.DataFrame(
        [{"Dimension": d["name"], "Weight": d["weight"]} for d in SCORING_DIMENSIONS]
    )

    st.dataframe(weights_df, use_container_width=True, hide_index=True)

    st.divider()

    use_synthetic = st.toggle("Use synthetic demo data", value=True)

    st.caption(
        "Keep this ON for demo mode. Turn it OFF when you want to upload actual RFP and solution documents."
    )

st.subheader("1. Input Documents")

if use_synthetic:
    rfp_text = clean_text(SYNTHETIC_RFP)
    solution_text = clean_text(SYNTHETIC_SOLUTION)
    st.success("Using embedded synthetic LSH RFP and solution response.")
else:
    col1, col2 = st.columns(2)

    with col1:
        rfp_file = st.file_uploader(
            "Upload RFP",
            type=["pdf", "pptx", "docx", "txt"]
        )

    with col2:
        solution_file = st.file_uploader(
            "Upload solution deck / response",
            type=["pdf", "pptx", "docx", "txt"]
        )

    if not rfp_file or not solution_file:
        st.warning("Upload both files or turn ON synthetic demo data.")
        st.stop()

    with st.spinner("Parsing uploaded files..."):
        rfp_text = clean_text(extract_text(rfp_file))
        solution_text = clean_text(extract_text(solution_file))

st.subheader("2. Deal Readiness Score")

scorecard, total_score, rating, verdict, top_gaps = score_deal(rfp_text, solution_text)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Score", f"{total_score} / 100")
m2.metric("Readiness", rating)
m3.metric("Dimensions", len(scorecard))
m4.metric("Lowest Gap Score", f"{top_gaps.iloc[0]['Score / 5']} / 5")

if rating == "Green":
    st.success(verdict)
elif rating == "Amber":
    st.warning(verdict)
else:
    st.error(verdict)

st.subheader("3. Weighted Scorecard")
st.dataframe(scorecard, use_container_width=True, hide_index=True)

st.subheader("4. Top Improvement Areas")

for _, row in top_gaps.iterrows():
    st.markdown(f"**{row['Dimension']}** — `{row['Score / 5']} / 5`")
    st.write(row["Feedback"])

st.subheader("5. Deal Heatmap")

chart_df = scorecard[["Dimension", "Weighted Score"]].set_index("Dimension")
st.bar_chart(chart_df)

st.subheader("6. Text Preview")

with st.expander("RFP preview"):
    st.write(rfp_text[:7000])

with st.expander("Solution response preview"):
    st.write(solution_text[:7000])

st.subheader("7. Export Scorecard")

output = io.BytesIO()

with pd.ExcelWriter(output, engine="openpyxl") as writer:
    scorecard.to_excel(writer, index=False, sheet_name="Scorecard")
    pd.DataFrame(
        [
            {
                "Total Score": total_score,
                "Rating": rating,
                "Verdict": verdict,
            }
        ]
    ).to_excel(writer, index=False, sheet_name="Summary")
    top_gaps.to_excel(writer, index=False, sheet_name="Top Gaps")

st.download_button(
    label="Download Excel Scorecard",
    data=output.getvalue(),
    file_name="lsh_deal_analysis_scorecard.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
