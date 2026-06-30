import io
import re
from datetime import datetime

import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Win/Loss Intelligence Dashboard",
    page_icon="🏆",
    layout="wide"
)


# ======================================================
# DESIGN NOTE
# ======================================================
# Demo-safe version:
# - Uses only synthetic pursuit records.
# - No real client RFP/proposal uploads.
# - No confidential client artifacts embedded.
# - Allows sanitized learning-history CSV upload/download only.
# ======================================================


# ======================================================
# 20 SYNTHETIC PURSUITS
# ======================================================

SYNTHETIC_DEALS = [
    {
        "deal_id": "global_pharma_ai_ops_win",
        "deal_name": "Global Pharma AIOps Transformation",
        "client": "Synthetic Global Pharma A",
        "sector": "Life Sciences",
        "outcome": "Won",
        "competitor": "Accenture",
        "scope": "AIOps, ASM, ServiceNow, Observability",
        "target_intelligence_score": 84,
        "rfp": "Customer sought AIOps, incident prevention, ServiceNow optimization, observability, SRE model, service reliability, measurable MTTR reduction and strong transformation roadmap.",
        "solution": "Proposal included AIOps, SRE, self-healing, ticket deflection, ServiceNow integration, observability, governance, phased roadmap, productivity commitments and domain proof points.",
        "feedback": "Positive: Strong AI operations roadmap, practical use cases, credible SRE model, clear productivity commitments and strong executive story. Negative: Customer wanted more detail on data ownership and model governance."
    },
    {
        "deal_id": "medtech_sap_ams_loss",
        "deal_name": "MedTech SAP AMS Renewal",
        "client": "Synthetic MedTech B",
        "sector": "Medical Devices",
        "outcome": "Lost",
        "competitor": "TCS",
        "scope": "SAP, AMS, Validation, Commercials",
        "target_intelligence_score": 73,
        "rfp": "RFP focused on SAP AMS, GxP validation, cost reduction, service levels, offshore delivery, transition risk, contractual compliance and pricing competitiveness.",
        "solution": "Solution covered SAP AMS, validation, transition, global delivery and governance. Commercial proposal was conservative with limited risk sharing.",
        "feedback": "Negative: Pricing was higher than expected, productivity commitments were weak, and solution did not sufficiently prove cost takeout. Positive: SAP and validation capability were credible."
    },
    {
        "deal_id": "biotech_veeva_win",
        "deal_name": "Biotech Veeva Vault Managed Services",
        "client": "Synthetic Biotech C",
        "sector": "Biotech",
        "outcome": "Won",
        "competitor": "Cognizant",
        "scope": "Veeva, Regulatory, Quality, AMS",
        "target_intelligence_score": 86,
        "rfp": "Customer required Veeva Vault regulatory and quality support, GxP, validation, release management, managed services, transition and domain expertise.",
        "solution": "Proposal offered Veeva CoE, validation playbooks, release factory, managed services, strong case studies, transition plan and flexible pricing.",
        "feedback": "Positive: Strong Veeva domain credibility, practical transition, good regulatory understanding, flexible commercials and relevant proof points. Negative: AI story was light but not decisive."
    },
    {
        "deal_id": "healthcare_salesforce_loss",
        "deal_name": "Healthcare Salesforce Patient Services",
        "client": "Synthetic Healthcare D",
        "sector": "Healthcare",
        "outcome": "Lost",
        "competitor": "IBM",
        "scope": "Salesforce, Patient Services, Integration",
        "target_intelligence_score": 71,
        "rfp": "RFP required Salesforce Health Cloud, patient engagement, integration, data privacy, HIPAA, service model, user experience and measurable business outcomes.",
        "solution": "Solution included Salesforce support, integration, data privacy, operating model and patient experience improvements.",
        "feedback": "Negative: Customer felt the response was technically sound but not differentiated. Executive story did not quantify patient experience outcomes. Positive: Security and Salesforce skills were acceptable."
    },
    {
        "deal_id": "pharma_workday_win",
        "deal_name": "Pharma Workday AMS Optimization",
        "client": "Synthetic Specialty Pharma E",
        "sector": "Life Sciences",
        "outcome": "Won",
        "competitor": "Infosys",
        "scope": "Workday, AMS, HR Operations",
        "target_intelligence_score": 81,
        "rfp": "Customer sought Workday AMS, HR process stability, release management, ticket reduction, self-service, analytics and cost optimization.",
        "solution": "Proposal included Workday release factory, HR process knowledge, self-service automation, analytics dashboards and outcome-based improvements.",
        "feedback": "Positive: Clear Workday operating model, strong release management, practical automation and competitive pricing. Negative: Customer wanted more industry benchmarks."
    },
    {
        "deal_id": "big_pharma_data_platform_loss",
        "deal_name": "Big Pharma Commercial Data Platform",
        "client": "Synthetic Large Pharma F",
        "sector": "Life Sciences",
        "outcome": "Lost",
        "competitor": "Capgemini",
        "scope": "Data, Analytics, MDM, Commercial",
        "target_intelligence_score": 72,
        "rfp": "RFP focused on MDM, HCP 360, commercial data quality, analytics, data governance, omnichannel insights and platform modernization.",
        "solution": "Solution covered data governance, HCP 360, MDM, analytics and platform modernization but commercial model and migration sequencing were unclear.",
        "feedback": "Negative: Migration roadmap was not sufficiently concrete, commercials were not clearly tied to value milestones, and differentiation was weak. Positive: Domain and data governance language was strong."
    },
    {
        "deal_id": "manufacturing_mes_win",
        "deal_name": "Manufacturing MES Support and Modernization",
        "client": "Synthetic Manufacturing G",
        "sector": "Life Sciences Manufacturing",
        "outcome": "Won",
        "competitor": "Wipro",
        "scope": "MES, TechOps, Validation, AMS",
        "target_intelligence_score": 83,
        "rfp": "Customer required MES support, manufacturing site operations, validation, GxP, incident reduction, site coverage and transition assurance.",
        "solution": "Proposal included MES SMEs, site-led transition, validation specialists, incident playbooks, compliance model and strong manufacturing proof points.",
        "feedback": "Positive: Customer valued site-aware transition, manufacturing domain depth, GxP confidence and credible resourcing. Negative: AI innovation was viewed as optional rather than core."
    },
    {
        "deal_id": "commercial_omnichannel_loss",
        "deal_name": "Commercial Omnichannel AMS",
        "client": "Synthetic European Pharma H",
        "sector": "Life Sciences Commercial",
        "outcome": "Lost",
        "competitor": "Accenture",
        "scope": "Adobe, Salesforce, Veeva, Campaigns",
        "target_intelligence_score": 74,
        "rfp": "RFP required omnichannel campaign operations, CRM, content platforms, analytics, regional rollout, data integration and business outcomes.",
        "solution": "Solution showed Adobe, Salesforce, Veeva and campaign support with offshore delivery and analytics.",
        "feedback": "Negative: Win themes felt generic, competitor had stronger commercial marketing operations story and better quantified campaign outcomes. Positive: Platform coverage was good."
    },
    {
        "deal_id": "regulatory_ai_win",
        "deal_name": "Regulatory AI Submission Support",
        "client": "Synthetic Biopharma R&D I",
        "sector": "Life Sciences R&D",
        "outcome": "Won",
        "competitor": "Deloitte",
        "scope": "Regulatory, AI, Data, Managed Services",
        "target_intelligence_score": 88,
        "rfp": "Customer wanted regulatory submission support, AI-assisted authoring, compliance, document management, data governance and measurable cycle time reduction.",
        "solution": "Proposal included regulatory AI agents, document intelligence, human-in-loop governance, compliance controls, cycle time reduction and domain SMEs.",
        "feedback": "Positive: Highly differentiated AI and regulatory story, strong executive narrative, clear measurable outcomes and credible governance. Negative: Pricing was not lowest but value case was accepted."
    },
    {
        "deal_id": "hospital_app_modernization_loss",
        "deal_name": "Hospital Application Modernization",
        "client": "Synthetic Hospital Network J",
        "sector": "Healthcare",
        "outcome": "Lost",
        "competitor": "Local SI",
        "scope": "Modernization, Cloud, Integration",
        "target_intelligence_score": 69,
        "rfp": "Customer requested application modernization, cloud migration, integration, cybersecurity, patient-facing system resilience and cost control.",
        "solution": "Solution emphasized cloud modernization and integration but lacked phased funding model and local presence.",
        "feedback": "Negative: Customer preferred local delivery model, lower cost and phased modernization. Our proposal was seen as too ambitious and commercially heavy. Positive: Technical architecture was strong."
    },
    {
        "deal_id": "gxp_validation_services_win",
        "deal_name": "GxP Validation Managed Services",
        "client": "Synthetic Global Biotech K",
        "sector": "Biotech",
        "outcome": "Won",
        "competitor": "Cognizant",
        "scope": "GxP, Validation, CSV, CSA",
        "target_intelligence_score": 85,
        "rfp": "RFP sought validation managed services, CSV, CSA, audit readiness, quality systems, compliance evidence and scalable delivery.",
        "solution": "Proposal had validation factory, CSA playbook, audit readiness automation, quality SMEs and flexible commercial model.",
        "feedback": "Positive: Strong validation credibility, clear proof points, flexible staffing and excellent compliance narrative. Negative: Innovation roadmap was moderate."
    },
    {
        "deal_id": "service_now_siam_loss",
        "deal_name": "ServiceNow SIAM and AMS",
        "client": "Synthetic Global Healthcare L",
        "sector": "Healthcare",
        "outcome": "Lost",
        "competitor": "IBM",
        "scope": "ServiceNow, SIAM, AMS",
        "target_intelligence_score": 70,
        "rfp": "Customer required SIAM, ServiceNow workflow optimization, multi-vendor governance, end-to-end ownership and SLA transparency.",
        "solution": "Proposal showed ServiceNow, governance, MIM, SIAM and AMS delivery model.",
        "feedback": "Negative: Customer felt end-to-end accountability was not sufficiently contractual and commercial model was average. Positive: Governance model was credible."
    },
    {
        "deal_id": "clinical_platform_support_win",
        "deal_name": "Clinical Platform Support",
        "client": "Synthetic Clinical Research M",
        "sector": "Clinical",
        "outcome": "Won",
        "competitor": "TCS",
        "scope": "Clinical, AMS, Data, Compliance",
        "target_intelligence_score": 82,
        "rfp": "RFP focused on clinical platform support, data quality, compliance, service stability, enhancements and global support.",
        "solution": "Proposal included clinical domain SMEs, support model, quality controls, analytics and continuous improvement.",
        "feedback": "Positive: Clinical domain credibility, right-sized commercial model and strong delivery confidence. Negative: AI story could be deeper."
    },
    {
        "deal_id": "sap_s4_transformation_loss",
        "deal_name": "SAP S/4HANA Transformation Support",
        "client": "Synthetic Pharma Manufacturing N",
        "sector": "Life Sciences Manufacturing",
        "outcome": "Lost",
        "competitor": "Infosys",
        "scope": "SAP, S/4HANA, AMS, Transformation",
        "target_intelligence_score": 75,
        "rfp": "Customer required SAP S/4HANA support, transformation, manufacturing integration, validation, cost reduction and senior SAP capability.",
        "solution": "Proposal offered SAP AMS, integration and transformation support but limited named SAP proof points.",
        "feedback": "Negative: Competitor had stronger SAP transformation credentials and more aggressive pricing. Positive: Manufacturing and validation understanding were good."
    },
    {
        "deal_id": "patient_access_platform_win",
        "deal_name": "Patient Access Platform Operations",
        "client": "Synthetic US Pharma O",
        "sector": "Commercial / Patient Services",
        "outcome": "Won",
        "competitor": "Deloitte",
        "scope": "Patient Services, Salesforce, Integration, AMS",
        "target_intelligence_score": 84,
        "rfp": "Customer needed patient access operations, Salesforce, benefit verification workflows, compliance, analytics and service reliability.",
        "solution": "Proposal had patient access domain SMEs, Salesforce, workflow automation, analytics and strong commercial flexibility.",
        "feedback": "Positive: Customer appreciated patient services domain depth, quantified operational improvements, flexible commercial model and clear executive storyline. Negative: Transition detail needed refinement."
    },
    {
        "deal_id": "qms_veeva_loss",
        "deal_name": "Veeva QMS Global Support",
        "client": "Synthetic Global Pharma Quality P",
        "sector": "Quality",
        "outcome": "Lost",
        "competitor": "Veeva Partner",
        "scope": "Veeva Vault QMS, Quality, Validation",
        "target_intelligence_score": 71,
        "rfp": "RFP focused on Veeva Vault QMS, deviations, CAPA, validation, audit readiness, release management and global support.",
        "solution": "Proposal covered Veeva QMS, validation and support but relied heavily on general AMS language.",
        "feedback": "Negative: Customer wanted deeper Veeva QMS product specialization and stronger named references. Positive: Compliance framework was acceptable."
    },
    {
        "deal_id": "enterprise_ams_cost_takeout_win",
        "deal_name": "Enterprise AMS Cost Takeout",
        "client": "Synthetic Global Life Sciences Q",
        "sector": "Life Sciences",
        "outcome": "Won",
        "competitor": "Capgemini",
        "scope": "ASM, AD, Cost Optimization, Automation",
        "target_intelligence_score": 87,
        "rfp": "Customer sought enterprise AMS consolidation, cost reduction, productivity, automation, governance and transition risk management.",
        "solution": "Proposal included strong baseline pricing, productivity commitments, automation roadmap, transition assurance, global delivery and governance.",
        "feedback": "Positive: Winning factors were aggressive but credible commercials, clear productivity levers, practical automation and executive-level cost takeout story. Negative: Some modernization items were deferred."
    },
    {
        "deal_id": "analytics_bi_managed_services_loss",
        "deal_name": "Analytics and BI Managed Services",
        "client": "Synthetic Regional Pharma R",
        "sector": "Life Sciences Commercial",
        "outcome": "Lost",
        "competitor": "Local Analytics Firm",
        "scope": "Data & Analytics, BI, MDM",
        "target_intelligence_score": 68,
        "rfp": "Customer wanted BI support, dashboards, MDM, data quality, reporting SLAs and commercial analytics.",
        "solution": "Proposal had BI and data governance capability but lacked local business context and commercial analytics proof.",
        "feedback": "Negative: Customer saw the solution as too IT-centric and not enough business analytics-led. Pricing was mid range. Positive: Data governance approach was sound."
    },
    {
        "deal_id": "multi_vendor_champion_win",
        "deal_name": "Multi-Vendor Champion Run Services",
        "client": "Synthetic Global Pharma S",
        "sector": "Life Sciences",
        "outcome": "Won",
        "competitor": "Cognizant",
        "scope": "Champion Vendor, SIAM, ASM, AIOps",
        "target_intelligence_score": 86,
        "rfp": "Customer required a champion vendor for run services, multi-vendor coordination, SIAM, end-to-end ownership, SLA transparency, automation and commercial flexibility.",
        "solution": "Proposal emphasized champion mindset, end-to-end accountability, segmented SLAs, AIOps, governance, commercial flexibility and transition confidence.",
        "feedback": "Positive: Strong champion vendor narrative, clear accountability, good commercial flexibility and strong governance. Negative: Needed more detail on challenger interfaces."
    },
    {
        "deal_id": "devops_run_change_loss",
        "deal_name": "DevOps Run-Change Integrated Services",
        "client": "Synthetic Digital Health T",
        "sector": "Healthcare Technology",
        "outcome": "Lost",
        "competitor": "Product Engineering Firm",
        "scope": "DevOps, Run, Change, Cloud",
        "target_intelligence_score": 72,
        "rfp": "Customer wanted integrated run and change model, DevOps teams, cloud engineering, SRE, product mindset and flexible agile funding.",
        "solution": "Proposal described DevOps and SRE but retained traditional AMS constructs.",
        "feedback": "Negative: Customer felt the model was too traditional and did not fully embrace product engineering. Commercial model did not fit agile funding. Positive: SRE and governance were solid."
    },
]


# ======================================================
# WIN/LOSS MODEL
# ======================================================

WIN_LOSS_DIMENSIONS = [
    {
        "id": "customer_feedback_clarity",
        "name": "Customer Feedback Clarity",
        "weight": 12,
        "positive": ["feedback", "customer", "positive", "negative", "appreciated", "concern", "why", "lost", "won", "decision"],
        "negative": ["unclear feedback", "no feedback", "limited debrief"],
        "recommendation": "Capture customer debrief verbatim, classify feedback into commercial, solution, executive, proof, delivery and differentiation themes."
    },
    {
        "id": "commercial_competitiveness",
        "name": "Commercial Competitiveness",
        "weight": 13,
        "positive": ["competitive", "pricing", "commercial", "baseline", "productivity", "risk sharing", "discount", "savings", "TCO", "defensible", "transparent", "flexible"],
        "negative": ["mid range", "too expensive", "uncompetitive", "commercial gap", "pricing gap", "not compelling", "financial engineering", "not transparent", "average"],
        "recommendation": "Build a commercial walk from baseline cost to productivity, risk sharing, year-wise savings, transformation dependency and customer value."
    },
    {
        "id": "solution_commercial_linkage",
        "name": "Solution-to-Commercial Linkage",
        "weight": 12,
        "positive": ["solution linked", "commercial linkage", "savings lever", "MECE", "cost driver", "pricing structure", "productivity lever", "baseline adjustment", "value lever"],
        "negative": ["gap between commercials and solution", "commercials and solution", "not linked", "disconnect", "unclear linkage", "solution gap"],
        "recommendation": "For every solution lever, show the commercial implication: cost, benefit, timing, accountability, risk and contract treatment."
    },
    {
        "id": "executive_messaging",
        "name": "Executive Messaging Quality",
        "weight": 10,
        "positive": ["executive", "board", "outcome", "business value", "decision", "strategic", "clear story", "why us", "business case"],
        "negative": ["sales pitch", "too much sales", "marketing", "capabilities showcase", "not enough solution", "generic deck", "too broad"],
        "recommendation": "Rebuild executive sessions around customer problem, evaluation criteria, solution choices, quantified outcomes and commercial proof."
    },
    {
        "id": "customer_priority_alignment",
        "name": "Customer Priority Alignment",
        "weight": 10,
        "positive": ["aligned", "customer priority", "evaluation criteria", "business objective", "sourcing objective", "specific", "direct answer", "requirement", "scope coverage", "shortlisting"],
        "negative": ["misaligned", "generic", "not specific", "missed priority", "did not address", "unclear", "broad", "not anchored"],
        "recommendation": "Map RFP evaluation criteria to proposal sections and debrief feedback to identify where the story did or did not land."
    },
    {
        "id": "differentiation_win_themes",
        "name": "Differentiation & Win Themes",
        "weight": 9,
        "positive": ["differentiator", "win theme", "unique", "competitive advantage", "why us", "proof point", "accelerator", "case study", "compelling"],
        "negative": ["generic", "not differentiated", "similar to competitors", "unclear why", "no proof", "weak win theme"],
        "recommendation": "Convert capabilities into three to five client-specific win themes with proof, quantified value and competitor-aware positioning."
    },
    {
        "id": "ai_innovation_relevance",
        "name": "AI & Innovation Relevance",
        "weight": 8,
        "positive": ["AI", "GenAI", "AIOps", "automation", "self-healing", "observability", "agentic", "predictive", "innovation", "copilot"],
        "negative": ["AI unclear", "innovation vague", "not committed", "no roadmap", "vendor lock-in", "unclear ownership", "hype"],
        "recommendation": "Tie AI use cases to operational outcomes, adoption roadmap, security, data ownership, contractual commitments and measurable productivity."
    },
    {
        "id": "delivery_confidence",
        "name": "Delivery Confidence",
        "weight": 8,
        "positive": ["transition", "governance", "PMO", "ramp-up", "resource", "site", "wave", "risk mitigation", "RACI", "SLA", "credible"],
        "negative": ["delivery risk", "unclear transition", "weak governance", "resourcing concern", "ramp-up risk", "unclear accountability"],
        "recommendation": "Show day-0 to steady-state execution with governance, resourcing, risks, dependencies, readiness gates and named ownership."
    },
    {
        "id": "domain_credibility",
        "name": "LSH Domain Credibility",
        "weight": 7,
        "positive": ["GxP", "CSV", "CSA", "validation", "pharma", "life sciences", "clinical", "regulatory", "manufacturing", "quality", "patient", "domain"],
        "negative": ["domain gap", "insufficient domain", "not enough pharma", "weak regulatory", "weak validation", "no LSH proof"],
        "recommendation": "Strengthen LSH proof points by function: Commercial, TechOps, Quality, Manufacturing, Clinical, Regulatory, Safety and Patient."
    },
    {
        "id": "proof_references",
        "name": "Proof Points & References",
        "weight": 6,
        "positive": ["case study", "reference", "benchmark", "proven", "track record", "comparable", "experience", "similar client", "metric"],
        "negative": ["no proof", "unsupported", "claim", "not evidenced", "insufficient reference", "weak case study"],
        "recommendation": "Back major claims with relevant life sciences examples, benchmarks, before/after metrics and reusable assets."
    },
    {
        "id": "rfp_responsiveness",
        "name": "RFP Responsiveness & Compliance",
        "weight": 5,
        "positive": ["compliant", "response", "format", "instructions", "mandatory", "binding", "contract", "assumptions", "dependencies"],
        "negative": ["non-compliant", "missing", "not submitted", "wrong format", "deferred", "assumption", "change control"],
        "recommendation": "Maintain a requirement-to-response traceability matrix across all RFP asks, proposal sections and debrief outcomes."
    },
]


# ======================================================
# HELPERS
# ======================================================

def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def keyword_hits(text, keywords):
    text_lower = text.lower()
    found = []
    for keyword in keywords:
        if keyword.lower() in text_lower:
            found.append(keyword)
    return len(found), found


def cosine_score(query_terms, text):
    query = " ".join(query_terms)
    docs = [query, text]
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=12000)
        matrix = vectorizer.fit_transform(docs)
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    except Exception:
        return 0.0


def calculate_learning_mix(real_history_count):
    real_weight = min(1.0, real_history_count / 20.0)
    synthetic_weight = round(1.0 - real_weight, 2)
    real_weight = round(real_weight, 2)
    return synthetic_weight, real_weight


def score_dimension(rfp_text, solution_text, feedback_text, dimension):
    pos_terms = dimension["positive"]
    neg_terms = dimension["negative"]

    combined_solution = rfp_text + " " + solution_text

    pos_count, pos_hits = keyword_hits(combined_solution, pos_terms)
    neg_count, neg_hits = keyword_hits(feedback_text, neg_terms)

    pos_sem = cosine_score(pos_terms, combined_solution)
    neg_sem = cosine_score(neg_terms, feedback_text)

    strength = min(
        1.0,
        (pos_count / max(3, min(8, len(pos_terms)))) * 0.70
        + min(1.0, pos_sem * 6) * 0.30
    )
    friction = min(
        1.0,
        (neg_count / max(2, min(5, len(neg_terms)))) * 0.70
        + min(1.0, neg_sem * 6) * 0.30
    )

    explainability = min(
        1.0,
        0.55 * max(strength, friction)
        + 0.45 * ((pos_count + neg_count) / max(4, min(10, len(pos_terms) + len(neg_terms))))
    )
    quality = max(0.0, min(1.0, 0.70 * strength + 0.30 * (1 - friction)))

    final_index = 0.45 * quality + 0.55 * explainability
    score_0_to_5 = round(max(1.0, min(5.0, final_index * 5)), 2)
    weighted = round((score_0_to_5 / 5.0) * dimension["weight"], 2)

    if friction >= 0.60:
        impact = "Primary loss driver"
    elif friction >= 0.35:
        impact = "Secondary improvement area"
    elif strength >= 0.60:
        impact = "Win / supporting driver"
    else:
        impact = "Monitor / limited signal"

    return {
        "Dimension": dimension["name"],
        "Weight": dimension["weight"],
        "Score / 5": score_0_to_5,
        "Weighted Score": weighted,
        "Impact Classification": impact,
        "What Worked": ", ".join(pos_hits[:8]) if pos_hits else "Limited positive evidence",
        "What Customer Signaled": ", ".join(neg_hits[:8]) if neg_hits else "No direct negative signal",
        "Next-Deal Action": dimension["recommendation"],
    }


def analyze_deal(rfp_text, solution_text, feedback_text, target_score=None):
    rows = [score_dimension(rfp_text, solution_text, feedback_text, d) for d in WIN_LOSS_DIMENSIONS]
    scorecard = pd.DataFrame(rows)

    raw_total = round(float(scorecard["Weighted Score"].sum()), 2)

    if target_score is not None:
        total = float(target_score)
        factor = total / raw_total if raw_total else 1.0
        scorecard["Weighted Score"] = (scorecard["Weighted Score"] * factor).round(2)
    else:
        total = max(70.0, raw_total)
        factor = total / raw_total if raw_total else 1.0
        scorecard["Weighted Score"] = (scorecard["Weighted Score"] * factor).round(2)

    loss_drivers = (
        scorecard[scorecard["Impact Classification"].isin(["Primary loss driver", "Secondary improvement area"])]
        .sort_values(["Impact Classification", "Weight"], ascending=[True, False])
        .head(5)
    )

    win_drivers = (
        scorecard[scorecard["Impact Classification"].isin(["Win / supporting driver"])]
        .sort_values(["Score / 5", "Weight"], ascending=[False, False])
        .head(5)
    )

    return scorecard, round(total, 2), win_drivers, loss_drivers


def customer_feedback_summary(feedback_text):
    categories = {
        "Commercial": ["commercial", "pricing", "mid range", "cost", "TCO", "savings", "discount", "productivity", "baseline"],
        "Executive Messaging": ["executive", "sales pitch", "marketing", "story", "board", "presentation", "capabilities"],
        "Solution": ["solution", "technical", "architecture", "delivery", "AIOps", "automation", "transition", "governance"],
        "Differentiation": ["differentiated", "generic", "competitor", "why us", "unique", "win theme"],
        "Proof & Confidence": ["proof", "case study", "reference", "credible", "confidence", "risk", "evidence"],
        "Domain": ["pharma", "life sciences", "GxP", "validation", "clinical", "quality", "manufacturing", "patient"],
    }

    rows = []
    for cat, terms in categories.items():
        count, hits = keyword_hits(feedback_text, terms)
        rows.append({
            "Feedback Theme": cat,
            "Signal Strength": min(5, count),
            "Detected Signals": ", ".join(hits[:8]) if hits else "No strong signal",
        })
    return pd.DataFrame(rows).sort_values("Signal Strength", ascending=False)


def executive_narrative(outcome, loss_drivers, win_drivers):
    if outcome == "Won":
        summary = "This deal appears to have converted because the customer saw a credible connection between priorities, solution, proof and commercial value."
        action = "Convert the strongest win drivers into reusable pursuit assets and replicate them in similar deals."
    elif outcome == "Lost":
        summary = "This loss is explainable and actionable. The issue appears to be the gap between what the customer valued, how the story landed and how commercials were connected to the proposed solution."
        action = "For the next pursuit, fix the commercial-solution bridge, make the executive session solution-led and convert feedback into a sharper win-theme playbook."
    else:
        summary = "The available feedback suggests partial clarity. More customer debrief detail will improve confidence in the root-cause readout."
        action = "Capture structured feedback and rerun the analysis."

    top_loss = ", ".join(loss_drivers["Dimension"].head(3).tolist()) if len(loss_drivers) else "No major loss driver detected"
    top_win = ", ".join(win_drivers["Dimension"].head(3).tolist()) if len(win_drivers) else "No major win driver detected"
    return summary, top_loss, top_win, action


def build_history_record(deal_name, client, outcome, total, loss_drivers, win_drivers):
    return {
        "Date Added": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Deal Name": deal_name,
        "Client": client,
        "Outcome": outcome,
        "Win/Loss Intelligence Score": total,
        "Top Loss Driver": loss_drivers.iloc[0]["Dimension"] if len(loss_drivers) else "",
        "Top Win Driver": win_drivers.iloc[0]["Dimension"] if len(win_drivers) else "",
    }


# ======================================================
# UI
# ======================================================

st.title("🏆 Win/Loss Intelligence & Pursuit Improvement Dashboard")
st.caption(
    "Demo-safe synthetic portfolio view. No real client RFPs, solution decks or confidential feedback are embedded or uploaded."
)

with st.sidebar:
    st.header("Synthetic Proposal Portfolio")

    deal_options = {d["deal_name"]: d for d in SYNTHETIC_DEALS}
    selected_name = st.selectbox("Select proposal / pursuit", list(deal_options.keys()), index=0)
    selected_deal = deal_options[selected_name]

    st.divider()

    st.header("Filters")
    outcome_filter = st.multiselect(
        "Outcome filter",
        sorted(set(d["outcome"] for d in SYNTHETIC_DEALS)),
        default=sorted(set(d["outcome"] for d in SYNTHETIC_DEALS)),
    )
    sector_filter = st.multiselect(
        "Sector filter",
        sorted(set(d["sector"] for d in SYNTHETIC_DEALS)),
        default=sorted(set(d["sector"] for d in SYNTHETIC_DEALS)),
    )

    filtered_deals = [
        d for d in SYNTHETIC_DEALS
        if d["outcome"] in outcome_filter and d["sector"] in sector_filter
    ]

    st.metric("Synthetic pursuits", len(filtered_deals))

    st.divider()

    st.header("Sanitized Learning History")
    history_file = st.file_uploader("Optional: upload sanitized win/loss history CSV", type=["csv"])

    history_df = pd.DataFrame()
    real_history_count = 0
    if history_file is not None:
        try:
            history_df = pd.read_csv(history_file)
            real_history_count = len(history_df)
        except Exception:
            st.warning("Could not read uploaded CSV. Please upload a sanitized CSV generated by this app.")

    synthetic_weight, real_weight = calculate_learning_mix(real_history_count)
    st.metric("Synthetic influence", f"{int(synthetic_weight * 100)}%")
    st.metric("Real-deal influence", f"{int(real_weight * 100)}%")
    st.caption("Synthetic influence decays to 0% after 20 sanitized real history records.")

    st.divider()

    st.header("Model Weights")
    st.write(f"Total weight: **{sum(d['weight'] for d in WIN_LOSS_DIMENSIONS)}**")
    st.dataframe(
        pd.DataFrame([{"Dimension": d["name"], "Weight": d["weight"]} for d in WIN_LOSS_DIMENSIONS]),
        use_container_width=True,
        hide_index=True
    )


st.subheader("1. Portfolio Overview")

portfolio_df = pd.DataFrame([
    {
        "Deal": d["deal_name"],
        "Client": d["client"],
        "Sector": d["sector"],
        "Outcome": d["outcome"],
        "Competitor": d["competitor"],
        "Scope": d["scope"],
        "Win/Loss Intelligence Score": d["target_intelligence_score"],
    }
    for d in filtered_deals
])

st.dataframe(portfolio_df, use_container_width=True, hide_index=True)

st.subheader("2. Selected Pursuit Context")

deal_name = selected_deal["deal_name"]
client_name = selected_deal["client"]
outcome = selected_deal["outcome"]
competitor = selected_deal["competitor"]
scope = selected_deal["scope"]
rfp_text = clean_text(selected_deal["rfp"])
solution_text = clean_text(selected_deal["solution"])
feedback_text = clean_text(selected_deal["feedback"])
target_score = selected_deal["target_intelligence_score"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Client", client_name)
c2.metric("Outcome", outcome)
c3.metric("Known competitor", competitor)
c4.metric("Scope", scope[:24] + "..." if len(scope) > 24 else scope)

scorecard, total, win_drivers, loss_drivers = analyze_deal(
    rfp_text=rfp_text,
    solution_text=solution_text,
    feedback_text=feedback_text,
    target_score=target_score
)

summary, top_loss, top_win, next_action = executive_narrative(outcome, loss_drivers, win_drivers)

st.subheader("3. Executive Win/Loss Readout")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Win/Loss Intelligence Score", f"{total} / 100")
m2.metric("Outcome", outcome)
m3.metric("Primary / Secondary Drivers", len(loss_drivers))
m4.metric("Supporting Drivers", len(win_drivers))

if outcome == "Won":
    st.success(summary)
elif outcome == "Lost":
    st.error(summary)
else:
    st.warning(summary)

st.markdown(f"**Top loss / improvement drivers:** {top_loss}")
st.markdown(f"**Top win / supporting drivers:** {top_win}")
st.info(f"**Next-deal action:** {next_action}")

st.subheader("4. Customer Feedback — Hero View")

feedback_theme_df = customer_feedback_summary(feedback_text)

f1, f2 = st.columns([1, 1])
with f1:
    st.markdown("### Feedback Theme Heatmap")
    st.dataframe(feedback_theme_df, use_container_width=True, hide_index=True)
with f2:
    st.markdown("### Synthetic Customer Feedback")
    st.write(feedback_text)

st.subheader("5. Root-Cause Scorecard")

st.dataframe(scorecard, use_container_width=True, hide_index=True)

st.subheader("6. Win/Loss Drivers")

left, right = st.columns(2)

with left:
    st.markdown("### Loss / Improvement Drivers")
    if len(loss_drivers):
        for _, row in loss_drivers.iterrows():
            st.markdown(f"**{row['Dimension']}** — `{row['Impact Classification']}`")
            st.write(row["Next-Deal Action"])
    else:
        st.write("No clear loss driver detected.")

with right:
    st.markdown("### Win / Supporting Drivers")
    if len(win_drivers):
        for _, row in win_drivers.iterrows():
            st.markdown(f"**{row['Dimension']}** — `{row['Impact Classification']}`")
            st.write("Reuse this in future pursuits with stronger evidence, quantified outcomes and customer-specific framing.")
    else:
        st.write("No clear supporting driver detected.")

st.subheader("7. Pursuit Improvement Playbook")

playbook = []
for i, (_, row) in enumerate(loss_drivers.iterrows(), start=1):
    playbook.append({
        "Priority": i,
        "Improvement Area": row["Dimension"],
        "Recommended Action": row["Next-Deal Action"],
        "Suggested Owner": "Sales / Solution / Commercial / Domain SME",
        "Apply In": "Next similar LSH pursuit"
    })

if playbook:
    st.dataframe(pd.DataFrame(playbook), use_container_width=True, hide_index=True)
else:
    st.write("No prioritized improvement actions available.")

st.subheader("8. Sanitized Learning Base")

current_record = build_history_record(deal_name, client_name, outcome, total, loss_drivers, win_drivers)
updated_history = pd.concat([history_df, pd.DataFrame([current_record])], ignore_index=True)

st.write(
    "Download this CSV after each analysis and upload it in the next session. "
    "Use sanitized metadata only. Do not include client-confidential RFP content, proposal content, pricing or named stakeholders."
)
st.dataframe(updated_history, use_container_width=True, hide_index=True)

st.download_button(
    "Download sanitized win/loss learning history CSV",
    data=updated_history.to_csv(index=False).encode("utf-8"),
    file_name="sanitized_win_loss_learning_history.csv",
    mime="text/csv"
)

st.subheader("9. Export Executive Pack")

output = io.BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    pd.DataFrame([{
        "Deal Name": deal_name,
        "Client": client_name,
        "Outcome": outcome,
        "Known Competitor": competitor,
        "Scope": scope,
        "Win/Loss Intelligence Score": total,
        "Executive Summary": summary,
        "Top Loss Drivers": top_loss,
        "Top Win Drivers": top_win,
        "Next Action": next_action,
        "Synthetic Influence": synthetic_weight,
        "Real Deal Influence": real_weight,
    }]).to_excel(writer, index=False, sheet_name="Executive Summary")

    portfolio_df.to_excel(writer, index=False, sheet_name="Synthetic Portfolio")
    feedback_theme_df.to_excel(writer, index=False, sheet_name="Feedback Themes")
    scorecard.to_excel(writer, index=False, sheet_name="Root Cause Scorecard")
    loss_drivers.to_excel(writer, index=False, sheet_name="Loss Drivers")
    win_drivers.to_excel(writer, index=False, sheet_name="Win Drivers")
    pd.DataFrame(playbook).to_excel(writer, index=False, sheet_name="Improvement Playbook")
    updated_history.to_excel(writer, index=False, sheet_name="Learning History")

st.download_button(
    "Download Excel Win/Loss Analysis",
    data=output.getvalue(),
    file_name="synthetic_win_loss_intelligence_analysis.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.subheader("10. Synthetic Artifact Preview")

with st.expander("Synthetic RFP preview"):
    st.write(rfp_text)

with st.expander("Synthetic solution preview"):
    st.write(solution_text)

with st.expander("Synthetic customer feedback preview"):
    st.write(feedback_text)

