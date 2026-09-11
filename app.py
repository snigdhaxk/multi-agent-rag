import os
import re
import hashlib

import faiss
import numpy as np
import streamlit as st

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PaperPilot AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #f8f5f3; color: #2b1b1b; }
    .main .block-container { max-width: 1380px; padding-top: 1.5rem; padding-bottom: 3rem; }
    #MainMenu, footer { visibility: hidden; }
    header { background: transparent !important; }

    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #eadfdb; }
    [data-testid="stSidebar"] .block-container { padding: 1.7rem 1.2rem; }

    .brand { display: flex; align-items: center; gap: .7rem; margin-bottom: .25rem; }
    .brand-mark { width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; border-radius: 12px; background: #f4e5e2; color: #7b1e2b; font-size: 1.35rem; }
    .brand-name { font-size: 1.25rem; font-weight: 800; color: #2b1b1b; }
    .brand-caption { color: #8b7772; font-size: .75rem; margin: .2rem 0 1.5rem 3rem; }
    .sidebar-heading { color: #9b8882; font-size: .68rem; font-weight: 800; letter-spacing: .9px; text-transform: uppercase; margin: 1.4rem 0 .65rem; }
    .sidebar-note { background: #fbf8f6; border: 1px solid #eadfdb; border-radius: 13px; padding: .85rem; color: #75625d; font-size: .76rem; line-height: 1.5; }

    .hero { background: #ffffff; border: 1px solid #eadfdb; border-radius: 22px; padding: 2rem 2.2rem; margin-bottom: 1.2rem; box-shadow: 0 8px 25px rgba(31,50,81,.045); }
    .hero-kicker { color: #7b1e2b; font-size: .72rem; font-weight: 800; letter-spacing: 1px; text-transform: uppercase; margin-bottom: .65rem; }
    .hero-title { color: #2b1b1b; font-size: 2.35rem; line-height: 1.15; font-weight: 800; letter-spacing: -1.2px; margin-bottom: .55rem; }
    .hero-subtitle { color: #75625d; max-width: 720px; font-size: .98rem; line-height: 1.65; }
    .trust-note { margin-top: 1rem; color: #75625d; font-size: .78rem; }

    .panel { background: #ffffff; border: 1px solid #eadfdb; border-radius: 18px; padding: 1.25rem; margin-bottom: 1rem; box-shadow: 0 7px 22px rgba(31,50,81,.035); }
    .panel-title { color: #2b1b1b; font-size: 1.05rem; font-weight: 800; margin-bottom: .25rem; }
    .panel-caption { color: #8b7772; font-size: .82rem; line-height: 1.5; margin-bottom: 1rem; }

    .metric-card { background: #ffffff; border: 1px solid #eadfdb; border-radius: 16px; padding: 1rem; min-height: 104px; box-shadow: 0 7px 22px rgba(31,50,81,.035); }
    .metric-icon { font-size: 1.15rem; margin-bottom: .45rem; }
    .metric-value { color: #2b1b1b; font-size: 1.25rem; font-weight: 800; }
    .metric-label { color: #8b7772; font-size: .73rem; margin-top: .2rem; }

    .ready-pill { display: inline-flex; align-items: center; gap: .4rem; padding: .38rem .7rem; border-radius: 999px; background: #f7ece9; border: 1px solid #dfc5bf; color: #7b1e2b; font-size: .72rem; font-weight: 800; }
    .ready-dot { width: 7px; height: 7px; border-radius: 50%; background: #9b3d4b; }

    .answer-card { background: #ffffff; border: 1px solid #e3d3ce; border-left: 4px solid #7b1e2b; border-radius: 16px; padding: 1.35rem; margin-top: 1rem; box-shadow: 0 8px 25px rgba(31,50,81,.04); }
    .answer-heading { color: #7b1e2b; font-size: .72rem; font-weight: 800; letter-spacing: .8px; text-transform: uppercase; margin-bottom: .85rem; }

    .evidence-card { background: #fdfaf9; border: 1px solid #eadfdb; border-radius: 13px; padding: 1rem; margin: .65rem 0; }
    .evidence-label { color: #7b1e2b; font-size: .74rem; font-weight: 800; margin-bottom: .45rem; }
    .evidence-text { color: #554541; font-size: .84rem; line-height: 1.6; }
    .suggestion-title { color: #554541; font-size: .8rem; font-weight: 800; margin-bottom: .6rem; }
    .processing-note { background: #fff8f6; border: 1px solid #e3c9c2; border-left: 4px solid #7b1e2b; color: #7b1e2b; border-radius: 10px; padding: .75rem 1rem; margin: .8rem 0; font-size: .85rem; font-weight: 700; }

    .stButton > button { border-radius: 10px; border: 1px solid #d9c8c2; background: #ffffff; color: #344054; font-weight: 700; min-height: 2.5rem; transition: all .18s ease; }
    .stButton > button:hover { border-color: #b9828b; background: #fbf1ef; color: #7b1e2b; }
    .stButton > button[kind="primary"] { background: #7b1e2b; border-color: #7b1e2b; color: #ffffff; }
    .stButton > button[kind="primary"]:hover { background: #641722; border-color: #641722; color: #ffffff; }

    .stDownloadButton > button { width: 100%; border-radius: 10px; background: #7b1e2b; border: 1px solid #7b1e2b; color: #ffffff; font-weight: 800; min-height: 2.6rem; }
    .stDownloadButton > button:hover { background: #641722; color: #ffffff; }

    .stTabs [data-baseweb="tab-list"] { gap: .35rem; background: #f1e8e5; border-radius: 13px; padding: .3rem; border: 1px solid #e5d7d2; }
    .stTabs [data-baseweb="tab"] {
        color: #7b1e2b !important;
        background: transparent !important;
        font-weight: 800 !important;
        border-radius: 9px;
        height: 2.7rem;
        padding: 0 1.1rem;
        opacity: 1 !important;
    }
    .stTabs [data-baseweb="tab"],
    .stTabs [data-baseweb="tab"] *,
    .stTabs [data-baseweb="tab"] span,
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] div,
    .stTabs [data-baseweb="tab"] button {
        color: #7b1e2b !important;
        -webkit-text-fill-color: #7b1e2b !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #ffffff !important;
        color: #7b1e2b !important;
        box-shadow: 0 2px 7px rgba(31,50,81,.08);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] * {
        color: #7b1e2b !important;
        -webkit-text-fill-color: #7b1e2b !important;
        opacity: 1 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="false"],
    .stTabs [data-baseweb="tab"][aria-selected="false"] * {
        color: #7b1e2b !important;
        -webkit-text-fill-color: #7b1e2b !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    /* Keep every tab label readable at all times. */
    div[data-testid="stTabs"] [role="tablist"] {
        background: #f1e8e5 !important;
        border: 1px solid #e5d7d2 !important;
        border-radius: 13px !important;
        padding: .3rem !important;
        gap: .35rem !important;
    }
    div[data-testid="stTabs"] button[role="tab"],
    div[data-testid="stTabs"] button[role="tab"] > div,
    div[data-testid="stTabs"] button[role="tab"] > div > div,
    div[data-testid="stTabs"] button[role="tab"] span,
    div[data-testid="stTabs"] button[role="tab"] p {
        color: #7b1e2b !important;
        -webkit-text-fill-color: #7b1e2b !important;
        opacity: 1 !important;
        visibility: visible !important;
        font-weight: 800 !important;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] > div,
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] span,
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {
        color: #7b1e2b !important;
        -webkit-text-fill-color: #7b1e2b !important;
        opacity: 1 !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background: #7b1e2b !important; }

    /* FINAL TAB VISIBILITY OVERRIDE */
    div[data-testid="stTabs"] > div:first-child,
    div[data-testid="stTabs"] [role="tablist"],
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        background: #f1e8e5 !important;
        opacity: 1 !important;
    }
    div[data-testid="stTabs"] [role="tab"],
    div[data-testid="stTabs"] [role="tab"] *,
    div[data-testid="stTabs"] [data-baseweb="tab"],
    div[data-testid="stTabs"] [data-baseweb="tab"] *,
    div[data-testid="stTabs"] [data-baseweb="tab"] p,
    div[data-testid="stTabs"] [data-baseweb="tab"] span,
    div[data-testid="stTabs"] [data-baseweb="tab"] div {
        color: #7b1e2b !important;
        -webkit-text-fill-color: #7b1e2b !important;
        opacity: 1 !important;
        visibility: visible !important;
        text-shadow: none !important;
    }
    div[data-testid="stTabs"] [role="tab"][aria-selected="false"],
    div[data-testid="stTabs"] [role="tab"][aria-selected="false"] *,
    div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="false"],
    div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="false"] * {
        color: #7b1e2b !important;
        -webkit-text-fill-color: #7b1e2b !important;
        opacity: 1 !important;
    }

    .stTextInput input { background: #ffffff; color: #2b1b1b; border: 1px solid #d9c8c2; border-radius: 10px; min-height: 2.7rem; }
    .stTextInput input:focus { border-color: #7b1e2b; box-shadow: 0 0 0 1px #7b1e2b; }
    /* File uploader: neutral and clearly visible, while burgundy remains the accent color. */
    [data-testid="stFileUploader"] {
        background: #ffffff !important;
        border: 1px dashed #b9828b !important;
        border-radius: 15px;
        padding: .75rem;
    }
    [data-testid="stFileUploader"] section {
        background: #ffffff !important;
        border: 0 !important;
        padding: .25rem !important;
    }
    [data-testid="stFileUploader"] section > div,
    [data-testid="stFileUploader"] section > div > div {
        background: #ffffff !important;
    }
    [data-testid="stFileUploader"] button {
        background: #ffffff !important;
        color: #7b1e2b !important;
        border: 1px solid #7b1e2b !important;
        border-radius: 9px !important;
        font-weight: 800 !important;
        box-shadow: none !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background: #f8eeee !important;
        color: #641722 !important;
        border-color: #641722 !important;
    }
    [data-testid="stFileUploader"] button span,
    [data-testid="stFileUploader"] button p,
    [data-testid="stFileUploader"] button div {
        color: #7b1e2b !important;
    }
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] p {
        color: #554541 !important;
    }
    [data-testid="stFileUploader"] svg {
        fill: #7b1e2b !important;
        color: #7b1e2b !important;
    }
    [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] {
        color: #554541 !important;
    }
    .stAlert { border-radius: 12px; }
    .stMarkdown { color: #344054; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_NAME = "openai/gpt-oss-20b"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

MAX_EVIDENCE_ITEMS = 7
MAX_CHARS_PER_EVIDENCE = 800


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "pages": [],
    "chunks": [],
    "index": None,
    "embeddings": None,
    "processed_file_hash": None,
    "uploaded_filename": None,
    "last_question": "",
    "last_answer": "",
    "last_evidence": [],
    "last_summary": "",
    "report_text": "",
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand"><div class="brand-mark">⌘</div><div class="brand-name">PaperPilot</div></div>
        <div class="brand-caption">Research intelligence workspace</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="ready-pill"><span class="ready-dot"></span> Workspace ready</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown('<div class="sidebar-heading">Workspace flow</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="sidebar-step">
            <div class="step-number">1</div>
            <div class="step-text">
                Upload a research paper in PDF format.
            </div>
        </div>

        <div class="sidebar-step">
            <div class="step-number">2</div>
            <div class="step-text">
                The system extracts and indexes important sections.
            </div>
        </div>

        <div class="sidebar-step">
            <div class="step-number">3</div>
            <div class="step-text">
                Ask questions or generate a complete paper summary.
            </div>
        </div>

        <div class="sidebar-step">
            <div class="step-number">4</div>
            <div class="step-text">
                Download your research analysis report.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown('<div class="sidebar-heading">Built with</div>', unsafe_allow_html=True)

    st.caption("📄 PDF Extraction")
    st.caption("🔎 FAISS Vector Search")
    st.caption("🧠 Sentence Transformers")
    st.caption("⚡ Groq LLM")
    st.caption("🎨 Streamlit UI")

    st.markdown("---")

    st.caption("PaperPilot AI • Research Assistant")
    st.caption("Built for faster academic understanding")


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Research paper intelligence workspace</div>
        <div class="hero-title">Understand research with clarity.</div>
        <div class="hero-subtitle">
            Upload a paper, ask focused questions, explore supporting evidence,
            and create a structured report without manually searching every page.
        </div>
        <div class="trust-note">✓ Answers are grounded in retrieved sections of your uploaded paper.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def extract_pages_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        text = re.sub(r"\s+", " ", text).strip()

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    return pages


def create_chunks(pages):
    chunks = []

    for page_data in pages:
        page_number = page_data["page"]
        text = page_data["text"]

        start = 0

        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_number,
                    }
                )

            if end >= len(text):
                break

            start = end - CHUNK_OVERLAP

    return chunks


def build_faiss_index(chunks, embedding_model):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, embeddings


def classify_question(question):
    question_lower = question.lower()

    if any(
        word in question_lower
        for word in [
            "limitation",
            "limitations",
            "weakness",
            "drawback",
            "future work",
            "future scope",
            "improve",
            "improvement",
        ]
    ):
        return "limitations"

    if any(
        word in question_lower
        for word in [
            "method",
            "methodology",
            "architecture",
            "algorithm",
            "model",
            "approach",
            "technique",
        ]
    ):
        return "methodology"

    if any(
        word in question_lower
        for word in [
            "dataset",
            "data",
            "images",
            "samples",
            "training data",
        ]
    ):
        return "dataset"

    if any(
        word in question_lower
        for word in [
            "result",
            "performance",
            "accuracy",
            "dice",
            "iou",
            "evaluation",
            "outcome",
        ]
    ):
        return "results"

    if any(
        word in question_lower
        for word in [
            "problem",
            "objective",
            "purpose",
            "challenge",
            "why",
        ]
    ):
        return "problem"

    return "general"


def expand_query(question):
    category = classify_question(question)

    expansions = {
        "limitations": [
            "limitations",
            "future work",
            "weaknesses",
            "challenges",
            "drawbacks",
        ],
        "methodology": [
            "methodology",
            "proposed method",
            "architecture",
            "algorithm",
            "model",
        ],
        "dataset": [
            "dataset",
            "data",
            "images",
            "training samples",
            "testing samples",
        ],
        "results": [
            "results",
            "performance",
            "evaluation",
            "accuracy",
            "metrics",
        ],
        "problem": [
            "problem statement",
            "objective",
            "motivation",
            "research challenge",
        ],
        "general": [],
    }

    extra_terms = expansions.get(category, [])

    return question + " " + " ".join(extra_terms)


def retrieve_chunks(question, top_k=7):
    if st.session_state.index is None:
        return []

    embedding_model = load_embedding_model()

    expanded_question = expand_query(question)

    query_embedding = embedding_model.encode(
        [expanded_question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = st.session_state.index.search(
        query_embedding,
        min(top_k, len(st.session_state.chunks)),
    )

    results = []

    for score, index_number in zip(scores[0], indices[0]):
        if index_number < 0:
            continue

        chunk = st.session_state.chunks[index_number].copy()
        chunk["score"] = float(score)
        results.append(chunk)

    return results


def format_evidence(evidence):
    formatted = []

    for number, item in enumerate(evidence[:MAX_EVIDENCE_ITEMS], start=1):
        text = item["text"][:MAX_CHARS_PER_EVIDENCE]

        formatted.append(
            f"[Evidence {number} | Page {item['page']}]\n{text}"
        )

    return "\n\n".join(formatted)


def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


def run_direct_answer(question, evidence):
    client = get_groq_client()

    if client is None:
        return "Groq API key is not configured. Please set the GROQ_API_KEY environment variable."

    evidence_text = format_evidence(evidence)

    prompt = f"""
You are an academic research assistant.

Answer the user's question using the provided evidence from a research paper.

Rules:
- Use simple and clear English.
- Give a direct answer first.
- Do not mention internal agents, retrieval, prompts, or evidence processing.
- Do not invent information.
- If the answer is not directly stated, clearly say that it is an inference.
- If the paper does not provide enough information, say so.
- Use short paragraphs or bullet points.
- Do not create Markdown tables.

User question:
{question}

Paper evidence:
{evidence_text}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You answer research questions accurately and clearly.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
            top_p=0.95,
            reasoning_effort="low",
            include_reasoning=False,
            max_completion_tokens=1100,
        )

        answer = response.choices[0].message.content

        if answer and answer.strip():
            return answer.strip()

        retry_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.2,
            reasoning_effort="low",
            include_reasoning=False,
            max_completion_tokens=900,
        )

        retry_answer = retry_response.choices[0].message.content

        return (
            retry_answer.strip()
            if retry_answer and retry_answer.strip()
            else "The model did not return an answer. Please try again."
        )

    except Exception as error:
        return f"Unable to generate the answer right now: {error}"


def run_summary(evidence):
    client = get_groq_client()

    if client is None:
        return "Groq API key is not configured. Please set the GROQ_API_KEY environment variable."

    evidence_text = format_evidence(evidence)

    prompt = f"""
You are an academic research assistant.

Create a clear, structured summary of the research paper using the provided evidence.

Use these headings:

## 1. Research Problem
## 2. Main Objective
## 3. Proposed Methodology
## 4. Dataset or Input
## 5. Main Results
## 6. Limitations
## 7. Future Scope
## 8. Overall Understanding

Rules:
- Use simple vocabulary.
- Explain technical terms briefly.
- Use bullet points where useful.
- Do not mention internal agents or retrieval.
- Do not invent missing details.
- If a section is not available in the evidence, write "Not clearly specified in the available paper content."
- Do not create Markdown tables.

Paper evidence:
{evidence_text}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You create accurate and beginner-friendly academic summaries.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
            top_p=0.95,
            reasoning_effort="low",
            include_reasoning=False,
            max_completion_tokens=1700,
        )

        summary = response.choices[0].message.content

        if summary and summary.strip():
            return summary.strip()

        return "The model did not return a summary. Please try again."

    except Exception as error:
        return f"Unable to generate the summary right now: {error}"


def create_download_report():
    report_parts = []

    report_parts.append("# PaperPilot AI Research Report\n")

    if st.session_state.uploaded_filename:
        report_parts.append(
            f"**Uploaded Paper:** {st.session_state.uploaded_filename}\n"
        )

    if st.session_state.last_question:
        report_parts.append("## Question\n")
        report_parts.append(st.session_state.last_question)
        report_parts.append("\n")

    if st.session_state.last_answer:
        report_parts.append("## Answer\n")
        report_parts.append(st.session_state.last_answer)
        report_parts.append("\n")

    if st.session_state.last_summary:
        report_parts.append("## Paper Summary\n")
        report_parts.append(st.session_state.last_summary)
        report_parts.append("\n")

    if st.session_state.last_evidence:
        report_parts.append("## Supporting Evidence\n")

        for item in st.session_state.last_evidence:
            report_parts.append(
                f"- Page {item['page']}: {item['text']}\n"
            )

    return "\n".join(report_parts)


def display_evidence(evidence):
    if not evidence:
        st.info("No supporting evidence is available yet.")
        return

    for number, item in enumerate(evidence, start=1):
        st.markdown(
            f"""
            <div class="evidence-card">
                <div class="evidence-label">
                    📌 Evidence {number} · Page {item['page']}
                </div>
                <div class="evidence-text">
                    {item['text']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    """
    <div class="panel">
        <div class="panel-title">Upload your research paper</div>
        <div class="panel-caption">
            Start by adding a PDF. PaperPilot will prepare it for questions, summaries, and evidence exploration.
        </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload PDF document",
    type=["pdf"],
    label_visibility="visible",
    help="Select a research paper in PDF format. Maximum file size: 200 MB.",
)

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    if st.session_state.processed_file_hash != file_hash:
        processing_message = st.empty()
        processing_message.markdown(
            '<div class="processing-note">Preparing your research workspace…</div>',
            unsafe_allow_html=True,
        )
        pages = extract_pages_from_pdf(uploaded_file)
        chunks = create_chunks(pages)
        embedding_model = load_embedding_model()
        index, embeddings = build_faiss_index(
            chunks,
            embedding_model,
        )
        processing_message.empty()

        st.session_state.pages = pages
        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.embeddings = embeddings
        st.session_state.processed_file_hash = file_hash
        st.session_state.uploaded_filename = uploaded_file.name

        st.session_state.last_question = ""
        st.session_state.last_answer = ""
        st.session_state.last_evidence = []
        st.session_state.last_summary = ""
        st.session_state.report_text = ""

        st.success("Your paper is ready to explore.")


# ============================================================
# PAPER METRICS
# ============================================================

if st.session_state.index is not None:
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">📄</div>
                <div class="metric-value">{len(st.session_state.pages)}</div>
                <div class="metric-label">Pages processed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">🧩</div>
                <div class="metric-value">{len(st.session_state.chunks)}</div>
                <div class="metric-label">Text chunks indexed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric_col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-icon">🔎</div>
                <div class="metric-value">FAISS</div>
                <div class="metric-label">Semantic search</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric_col4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-icon">⚡</div>
                <div class="metric-value">AI Ready</div>
                <div class="metric-label">Research assistant</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# MAIN WORKSPACE
# ============================================================

if st.session_state.index is None:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">🚀 Your research workspace is waiting</div>
            <div class="panel-caption">
                Upload a PDF above to ask questions, generate summaries,
                and inspect supporting evidence.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    tab_question, tab_summary, tab_evidence = st.tabs(
        [
            "💬 Ask Questions",
            "📝 Paper Summary",
            "📚 Supporting Evidence",
        ]
    )

    # ========================================================
    # QUESTION TAB
    # ========================================================

    with tab_question:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">💬 Ask your paper anything</div>
                <div class="panel-caption">
                    Ask questions in normal language. The system retrieves
                    relevant sections and generates a direct answer.
                </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="suggestion-title">💡 Try one of these questions</div>',
            unsafe_allow_html=True,
        )

        if "question_input" not in st.session_state:
            st.session_state.question_input = st.session_state.last_question

        suggestion_col1, suggestion_col2, suggestion_col3 = st.columns(3)

        with suggestion_col1:
            sample_problem = st.button(
                "What problem does this paper solve?",
                use_container_width=True,
                key="sample_problem",
            )

        with suggestion_col2:
            sample_methodology = st.button(
                "What methodology was used?",
                use_container_width=True,
                key="sample_methodology",
            )

        with suggestion_col3:
            sample_limitations = st.button(
                "What are the limitations?",
                use_container_width=True,
                key="sample_limitations",
            )

        question = st.text_input(
            "Your question",
            key="question_input",
            placeholder="Example: What is the main contribution of this paper?",
            label_visibility="collapsed",
        )

        ask_button = st.button(
            "✨ Get AI Answer",
            type="primary",
            use_container_width=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # Use the clicked sample button's actual value in this same run.
        sample_question = None
        if sample_problem:
            sample_question = "What problem does this paper solve?"
        elif sample_methodology:
            sample_question = "What methodology was used?"
        elif sample_limitations:
            sample_question = "What are the limitations?"

        if sample_question is not None:
            question = sample_question

        should_run_question = ask_button or sample_question is not None

        if should_run_question and question.strip():
            question_status = st.empty()
            question_status.markdown(
                '<div class="processing-note">Retrieving relevant sections and preparing your answer…</div>',
                unsafe_allow_html=True,
            )
            evidence = retrieve_chunks(question, top_k=7)
            answer = run_direct_answer(question, evidence)
            question_status.empty()

            st.session_state.last_question = question
            st.session_state.last_answer = answer
            st.session_state.last_evidence = evidence
            st.session_state.report_text = create_download_report()

        if st.session_state.last_answer:
            st.markdown(
                """
                <div class="answer-card">
                    <div class="answer-heading">✨ Research Answer</div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(st.session_state.last_answer)

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            report_text = create_download_report()

            st.download_button(
                label="⬇️ Download Research Report",
                data=report_text,
                file_name="paperpilot_research_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

    # ========================================================
    # SUMMARY TAB
    # ========================================================

    with tab_summary:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">📝 Generate a paper summary</div>
                <div class="panel-caption">
                    Get a beginner-friendly overview of the research problem,
                    methodology, dataset, results, limitations, and future scope.
                </div>
            """,
            unsafe_allow_html=True,
        )

        summary_button = st.button(
            "🚀 Generate Full Summary",
            type="primary",
            use_container_width=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

        if summary_button:
            summary_status = st.empty()
            summary_status.markdown(
                '<div class="processing-note">Retrieving the main sections of the paper…</div>',
                unsafe_allow_html=True,
            )
            summary_evidence = retrieve_chunks(
                """
                research problem objective methodology dataset
                results limitations future work conclusion
                """,
                top_k=7,
            )

            summary_status.markdown(
                '<div class="processing-note">Preparing your structured summary…</div>',
                unsafe_allow_html=True,
            )
            summary = run_summary(summary_evidence)
            summary_status.empty()

            st.session_state.last_summary = summary
            st.session_state.last_evidence = summary_evidence
            st.session_state.report_text = create_download_report()

        if st.session_state.last_summary:
            st.markdown(
                """
                <div class="answer-card">
                    <div class="answer-heading">📘 Paper Summary</div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(st.session_state.last_summary)

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            report_text = create_download_report()

            st.download_button(
                label="⬇️ Download Summary Report",
                data=report_text,
                file_name="paperpilot_summary_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

    # ========================================================
    # EVIDENCE TAB
    # ========================================================

    with tab_evidence:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">📚 Supporting Evidence</div>
                <div class="panel-caption">
                    Inspect the paper sections used to generate the latest answer or summary.
                </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.last_evidence:
            display_evidence(st.session_state.last_evidence)
        else:
            st.info(
                "Ask a question or generate a summary to view supporting evidence."
            )

        st.markdown("</div>", unsafe_allow_html=True)
