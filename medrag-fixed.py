# ============================================================
#   MEDISCAN AI — MEDICAL LAB REPORT EXPLAINER
#   Architecture: Query Processing → Retrieval (RAG) → Generation → Post-Processing → Safety & Triage
#   Run:  streamlit run medrag.py
#   Deps: pip install streamlit openai pdfplumber pillow reportlab
# ============================================================

import json, re, time, datetime, io, base64
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI

# ── API CONFIG ───────────────────────────────────────────────
API_KEY  = st.secrets["OPENROUTER_API_KEY"]
BASE_URL = "https://openrouter.ai/api/v1"
MODEL    = "openai/gpt-4o"
client   = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ============================================================
#  PAGE CONFIG & CSS
# ============================================================
st.set_page_config(page_title="MediScan AI", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
    background: #0b0f1a !important; color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1100px !important; }
[data-testid="stSidebar"] { background: #0f1423 !important; border-right: 1px solid rgba(99,179,237,0.12) !important; }
[data-testid="stSidebar"] * { color: #cbd5e0 !important; }
.hero-wrap {
    display: flex; align-items: center; gap: 18px; padding: 2.4rem 2.8rem;
    background: linear-gradient(135deg, #0f1e3a 0%, #0b1628 60%, #0d1f3c 100%);
    border: 1px solid rgba(99,179,237,0.18); border-radius: 20px; margin-bottom: 2rem;
    position: relative; overflow: hidden;
}
.hero-wrap::before {
    content: ''; position: absolute; top: -60px; right: -60px; width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(99,179,237,0.10) 0%, transparent 70%);
    border-radius: 50%; pointer-events: none;
}
.hero-icon { font-size: 3.2rem; line-height: 1; flex-shrink: 0; }
.hero-title { font-family: 'DM Serif Display', serif !important; font-size: 2.4rem !important; font-weight: 400 !important; color: #e8f4fd !important; letter-spacing: -0.5px; line-height: 1.1; margin: 0 0 4px !important; }
.hero-sub { font-size: 0.95rem; color: #7fb3d3; margin: 0; }
.card { background: #111827; border: 1px solid rgba(99,179,237,0.12); border-radius: 16px; padding: 1.6rem 2rem; margin-bottom: 1.4rem; }
.card-title { font-size: 0.72rem; font-weight: 600; letter-spacing: 1.8px; text-transform: uppercase; color: #63b3ed; margin-bottom: 1rem; }
[data-testid="stFileUploader"] { background: #0d1626 !important; border: 1.5px dashed rgba(99,179,237,0.25) !important; border-radius: 14px !important; }
[data-testid="stTextArea"] textarea { background: #0d1626 !important; border: 1px solid rgba(99,179,237,0.18) !important; border-radius: 12px !important; color: #e2e8f0 !important; }
[data-testid="stTextInput"] input { background: #0d1626 !important; border: 1px solid rgba(99,179,237,0.22) !important; border-radius: 12px !important; color: #e2e8f0 !important; height: 52px !important; }
.stButton > button { background: linear-gradient(135deg, #2563eb, #1d4ed8) !important; color: #fff !important; border: none !important; border-radius: 12px !important; font-size: 0.95rem !important; font-weight: 500 !important; height: 52px !important; transition: all 0.2s ease !important; }
.stButton > button:hover { background: linear-gradient(135deg, #3b82f6, #2563eb) !important; transform: translateY(-1px) !important; }
.badge { display: inline-block; padding: 4px 14px; border-radius: 30px; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.8px; text-transform: uppercase; }
.badge-urgent  { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.badge-soon    { background: rgba(251,191,36,0.12); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.badge-routine { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.flag-chip { display: inline-block; background: rgba(239,68,68,0.1); color: #fca5a5; border: 1px solid rgba(239,68,68,0.25); border-radius: 8px; padding: 3px 10px; font-size: 0.8rem; margin: 3px 4px 3px 0; }
.result-box { background: #0d1626; border: 1px solid rgba(99,179,237,0.15); border-left: 3px solid #63b3ed; border-radius: 0 14px 14px 0; padding: 1.4rem 1.6rem; font-size: 0.97rem; line-height: 1.75; color: #cbd5e0; word-wrap: break-word; }
.result-box p { margin: 0.45rem 0; }
.result-box br + br { display: none; }
.alert-urgent { background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.3); border-radius: 12px; padding: 1rem 1.4rem; color: #fca5a5; margin-bottom: 1rem; font-size: 0.92rem; }
.alert-soon   { background: rgba(251,191,36,0.07); border: 1px solid rgba(251,191,36,0.3); border-radius: 12px; padding: 1rem 1.4rem; color: #fde68a; margin-bottom: 1rem; font-size: 0.92rem; }
.alert-info   { background: rgba(99,179,237,0.07); border: 1px solid rgba(99,179,237,0.25); border-radius: 12px; padding: 1rem 1.4rem; color: #93c5fd; margin-bottom: 1rem; font-size: 0.92rem; }
.step-row { display: flex; align-items: center; gap: 10px; padding: 6px 0; font-size: 0.88rem; }
.step-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.step-dot.done    { background: #34d399; }
.step-dot.active  { background: #63b3ed; animation: pulse-dot 1s ease infinite; }
.step-dot.waiting { background: #374151; }
@keyframes pulse-dot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(1.4)} }
.history-item { background: #0f1623; border: 1px solid rgba(99,179,237,0.1); border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.8rem; }
.history-q { font-size: 0.88rem; color: #93c5fd; margin-bottom: 4px; font-weight: 500; }
.history-preview { font-size: 0.8rem; color: #64748b; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.cat-pill { display: inline-block; background: rgba(99,179,237,0.1); color: #93c5fd; border: 1px solid rgba(99,179,237,0.25); border-radius: 20px; padding: 3px 14px; font-size: 0.82rem; font-weight: 500; }
.metric-box { flex: 1; background: #0d1626; border: 1px solid rgba(99,179,237,0.12); border-radius: 12px; padding: 0.9rem 1.1rem; text-align: center; }
.metric-val { font-size: 1.4rem; font-weight: 600; color: #e2e8f0; }
.metric-lbl { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-top: 2px; }
.disclaimer { background: rgba(99,179,237,0.05); border: 1px solid rgba(99,179,237,0.12); border-radius: 12px; padding: 0.9rem 1.2rem; font-size: 0.82rem; color: #64748b; text-align: center; margin-top: 1.2rem; }
.rag-badge { display:inline-block; background:rgba(52,211,153,0.1); color:#34d399; border:1px solid rgba(52,211,153,0.25); border-radius:8px; padding:2px 10px; font-size:0.75rem; margin-left:8px; vertical-align:middle; }
hr { border-color: rgba(99,179,237,0.1) !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0b0f1a; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 4px; }

.triage-chart-card {
    background: linear-gradient(135deg, rgba(13,22,38,0.96), rgba(17,24,39,0.96));
    border: 1px solid rgba(99,179,237,0.16);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    margin: 1rem 0 1.2rem 0;
    box-shadow: 0 0 0 1px rgba(99,179,237,0.03), 0 10px 30px rgba(0,0,0,0.22);
}
.triage-chart-title {
    font-size: 0.78rem;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #63b3ed;
    margin-bottom: 1rem;
    font-weight: 600;
}
.triage-scale {
    display: flex;
    gap: 10px;
    align-items: flex-end;
    justify-content: space-between;
    height: 180px;
    margin-top: 0.4rem;
}
.triage-bar-wrap {
    flex: 1;
    text-align: center;
}
.triage-bar-outer {
    height: 130px;
    width: 100%;
    max-width: 90px;
    margin: 0 auto 10px auto;
    background: rgba(148,163,184,0.08);
    border: 1px solid rgba(99,179,237,0.10);
    border-radius: 14px;
    display: flex;
    align-items: end;
    overflow: hidden;
    box-shadow: inset 0 0 20px rgba(0,0,0,0.18);
}
.triage-bar-fill {
    width: 100%;
    border-radius: 12px 12px 0 0;
    transition: height 0.5s ease;
    box-shadow: 0 0 24px rgba(99,179,237,0.18);
}
.triage-label {
    font-size: 0.78rem;
    color: #cbd5e0;
    font-weight: 500;
}
.triage-value {
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 3px;
}
.triage-current-note {
    margin-top: 0.9rem;
    padding: 0.75rem 0.95rem;
    border-radius: 12px;
    background: rgba(99,179,237,0.06);
    border: 1px solid rgba(99,179,237,0.12);
    color: #cbd5e0;
    font-size: 0.84rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
#  SESSION STATE
# ============================================================
for k, v in {
    "lab_report": "", "report_loaded": False, "chat_history": [],
    "test_category": "", "total_questions": 0, "flagged_count": 0,
    "chunks": [], "embeddings": []
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
#  LAYER 0 — FILE EXTRACTION
# ============================================================
def extract_from_upload(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        import pdfplumber
        text_parts = []
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return "\n\n".join(text_parts)
    elif any(name.endswith(e) for e in (".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp")):
        raw = uploaded_file.read()
        b64 = base64.b64encode(raw).decode("utf-8")
        ext = name.split(".")[-1].replace("jpg", "jpeg")
        mime = "image/" + ext
        response = client.chat.completions.create(
            model=MODEL, max_tokens=500,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:" + mime + ";base64," + b64}},
                {"type": "text", "text": (
                    "This is a medical lab report image. Extract ALL text exactly as it appears, "
                    "preserving test names, values, units, and reference ranges. "
                    "Return only the extracted text, no commentary."
                )}
            ]}]
        )
        return response.choices[0].message.content.strip()
    return ""


# ============================================================
#  LAYER 1 — QUERY PROCESSING
# ============================================================
def llm(system: str, user: str, max_tokens=400, temperature=0.3) -> str:
    r = client.chat.completions.create(
        model=MODEL, max_tokens=max_tokens, temperature=temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}]
    )
    return r.choices[0].message.content.strip()


def classify_category(report: str) -> str:
    """Test Category Classifier"""
    return llm(
        "You are a medical test classifier. Identify ONLY the test panel category "
        "(e.g. CBC, Lipid Panel, Comprehensive Metabolic Panel, Thyroid Panel, "
        "HbA1c/Diabetes, Liver Function Tests, Kidney Function Tests, Urinalysis, "
        "Coagulation Panel, Hormone Panel, or Other). Return ONLY the category name.",
        "Lab Report:\n" + report, max_tokens=60
    )


def normalize_terms(report: str) -> str:
    """Medical Term Normalizer"""
    return llm(
        "You are a medical terminology expert. Expand all abbreviations inline "
        "(e.g. WBC -> WBC (White Blood Cell Count), Hb -> Hemoglobin). "
        "Return the full report text with expansions, nothing else.",
        "Lab Report:\n" + report, max_tokens=800
    )


def rewrite_question(report_summary: str, category: str, question: str) -> str:
    """Patient-Friendly Query Rewriter"""
    return llm(
        "You are a medical query assistant for a " + category + " panel. "
        "Rewrite the patient's question into a precise medical query preserving the EXACT intent. "
        "If they ask about surgery keep that intent. If they ask what disease keep that intent. "
        "Return ONLY the rewritten question.",
        "Report summary:\n" + report_summary[:500] + "\n\nPatient question: " + question,
        max_tokens=200
    )


# ============================================================
#  LAYER 2 — RETRIEVAL (Embedding Model + Hybrid Retriever)
# ============================================================
def chunk_report(report: str) -> list:
    """Split report into meaningful chunks"""
    lines = [l.strip() for l in report.split("\n") if l.strip()]
    chunks, current = [], []
    for line in lines:
        current.append(line)
        if len(current) >= 5:
            chunks.append("\n".join(current))
            current = []
    if current:
        chunks.append("\n".join(current))
    return chunks if chunks else [report]


def get_embedding(text: str) -> list:
    """Embedding Model"""
    resp = client.embeddings.create(
        model="openai/text-embedding-ada-002",
        input=text[:2000]
    )
    return resp.data[0].embedding


def cosine_similarity(a: list, b: list) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x ** 2 for x in a) ** 0.5
    mag_b = sum(x ** 2 for x in b) ** 0.5
    return dot / (mag_a * mag_b + 1e-9)


def keyword_score(chunk: str, question: str) -> float:
    """Keyword component of hybrid retrieval"""
    stopwords = {"is","are","my","the","a","an","do","i","what","how","why","need","have","any","about"}
    words     = set(re.findall(r'\w+', question.lower())) - stopwords
    c_words   = set(re.findall(r'\w+', chunk.lower()))
    return len(words & c_words) / len(words) if words else 0.0


def hybrid_retrieve(chunks: list, embeddings: list, question: str, top_k: int = 5) -> str:
    """Hybrid Retriever: 70% semantic + 30% keyword"""
    if not chunks:
        return ""
    if not embeddings:
        # keyword-only fallback
        scores = sorted([(keyword_score(c, question), i) for i, c in enumerate(chunks)], reverse=True)
        return "\n\n---\n\n".join(chunks[i] for _, i in scores[:top_k])
    try:
        q_emb  = get_embedding(question)
        scores = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            hybrid = 0.7 * cosine_similarity(q_emb, emb) + 0.3 * keyword_score(chunk, question)
            scores.append((hybrid, i))
        scores.sort(reverse=True)
        return "\n\n---\n\n".join(chunks[i] for _, i in scores[:top_k])
    except Exception:
        scores = sorted([(keyword_score(c, question), i) for i, c in enumerate(chunks)], reverse=True)
        return "\n\n---\n\n".join(chunks[i] for _, i in scores[:top_k])


def build_index(report: str) -> tuple:
    """Build chunk index with embeddings"""
    chunks, embeddings = chunk_report(report), []
    try:
        for chunk in chunks:
            embeddings.append(get_embedding(chunk))
    except Exception:
        embeddings = []
    return chunks, embeddings


# ============================================================
#  LAYER 3 — GENERATION (LLM Generator)
# ============================================================
def generate_explanation(retrieved_context: str, question: str, category: str) -> str:
    """Question-specific answer grounded in retrieved context only"""
    return llm(
        "You are a friendly medical assistant explaining " + category + " lab results to a patient.\n"
        "You are given ONLY the most relevant sections of the lab report for this specific question.\n\n"
        "CRITICAL RULES:\n"
        "1. Answer SPECIFICALLY what the patient asked — never give a generic summary.\n"
        "2. Use ONLY the retrieved values provided below as your source.\n"
        "3. Handle question types as follows:\n"
        "   - 'Do I need surgery/treatment?' -> State if values suggest need for intervention, be direct.\n"
        "   - 'What disease do I have?' -> Name conditions the abnormal values suggest and explain why.\n"
        "   - 'What should I be worried about?' -> List only concerning findings with clear plain-English explanations.\n"
        "   - 'Are values abnormal?' -> For each value: state normal range, patient value, status (High/Low/Normal).\n"
        "   - 'Do I need a doctor urgently?' -> Answer Yes/No directly, then list exactly which values drive that answer.\n"
        "   - Specific value question -> Explain only that value: what it measures, normal range, patient level, meaning.\n"
        "4. Use plain language. Explain any medical term you use.\n"
        "5. End with exactly: 'Please consult your doctor for a proper diagnosis and treatment plan.'",
        "Retrieved Lab Context:\n" + retrieved_context + "\n\nPatient's Question: " + question + "\n\nAnswer specifically:",
        max_tokens=1200, temperature=0.35
    )


# ============================================================
#  LAYER 4 — POST-PROCESSING (Flag Detector + Confidence Score)
# ============================================================
def detect_flags(retrieved_context: str) -> dict:
    """Abnormal Value / Flag Detector + Confidence Score Generator"""
    raw = llm(
        "You are a medical flag detector. Analyze the lab values and return ONLY valid JSON with these keys:\n"
        "- abnormal_values: list of short strings like 'TestName: value (H/L)' for abnormal results only\n"
        "- severity: 'LOW' | 'MODERATE' | 'HIGH'\n"
        "- confidence: 'HIGH' | 'MEDIUM' | 'LOW'\n"
        "- needs_doctor_urgently: true | false\n"
        "Return ONLY the JSON object, no markdown, no explanation.",
        "Lab Values:\n" + retrieved_context, max_tokens=400, temperature=0.1
    )
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"abnormal_values": [], "severity": "LOW", "confidence": "MEDIUM", "needs_doctor_urgently": False}


# ============================================================
#  LAYER 5 — SAFETY & TRIAGE (Content Filter + Red-Flag Detection)
# ============================================================
def content_filter(explanation: str) -> str:
    """Content Filter"""
    return llm(
        "You are a medical content safety reviewer. Review this explanation and:\n"
        "1. Replace definitive diagnoses with 'may suggest' or 'could indicate'\n"
        "2. Remove specific medication names or dosage recommendations\n"
        "3. Soften fear-inducing language while keeping clinical accuracy\n"
        "4. Fix any factual errors about reference ranges\n"
        "If already safe, return unchanged. Return ONLY the reviewed explanation text.",
        "Explanation:\n" + explanation, max_tokens=1200, temperature=0.1
    )


def red_flag_triage(flags: dict) -> str:
    """Red-Flag Detection -> triage level"""
    severity = flags.get("severity", "LOW")
    urgent   = flags.get("needs_doctor_urgently", False)
    if urgent or severity == "HIGH":
        return "URGENT"
    elif severity == "MODERATE":
        return "SOON"
    return "ROUTINE"


# ============================================================
#  FULL PIPELINE
# ============================================================


def render_triage_chart(triage: str, confidence: str):
    triage_heights = {
        "ROUTINE": 28,
        "SOON": 62,
        "URGENT": 96
    }

    triage_colors = {
        "ROUTINE": "linear-gradient(180deg, #34d399, #15803d)",
        "SOON": "linear-gradient(180deg, #fbbf24, #d97706)",
        "URGENT": "linear-gradient(180deg, #fb7185, #dc2626)"
    }

    confidence_heights = {
        "LOW": 35,
        "MEDIUM": 68,
        "HIGH": 96
    }

    confidence_colors = {
        "LOW": "linear-gradient(180deg, #f87171, #b91c1c)",
        "MEDIUM": "linear-gradient(180deg, #fbbf24, #d97706)",
        "HIGH": "linear-gradient(180deg, #60a5fa, #2563eb)"
    }

    current_height = triage_heights.get(triage, 28)
    current_color = triage_colors.get(triage, "linear-gradient(180deg, #34d399, #15803d)")
    conf_height = confidence_heights.get(confidence, 68)
    conf_color = confidence_colors.get(confidence, "linear-gradient(180deg, #fbbf24, #d97706)")

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
        background: transparent;
        font-family: 'DM Sans', 'Segoe UI', sans-serif;
        color: #e2e8f0;
    }}
    .triage-chart-card {{
        background: linear-gradient(135deg, rgba(13,22,38,0.96), rgba(17,24,39,0.96));
        border: 1px solid rgba(99,179,237,0.16);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        margin: 0.4rem 0 0.6rem 0;
        box-shadow: 0 0 0 1px rgba(99,179,237,0.03), 0 10px 30px rgba(0,0,0,0.22);
    }}
    .triage-chart-title {{
        font-size: 0.78rem;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        color: #63b3ed;
        margin-bottom: 1rem;
        font-weight: 600;
    }}
    .triage-scale {{
        display: flex;
        gap: 10px;
        align-items: flex-end;
        justify-content: space-between;
        height: 160px;
        margin-top: 0.4rem;
    }}
    .triage-bar-wrap {{
        flex: 1;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
    }}
    .triage-bar-outer {{
        height: 110px;
        width: 100%;
        max-width: 90px;
        margin: 0 auto 8px auto;
        background: rgba(148,163,184,0.08);
        border: 1px solid rgba(99,179,237,0.10);
        border-radius: 14px;
        display: flex;
        align-items: flex-end;
        overflow: hidden;
        box-shadow: inset 0 0 20px rgba(0,0,0,0.18);
    }}
    .triage-bar-fill {{
        width: 100%;
        border-radius: 12px 12px 0 0;
        transition: height 0.5s ease;
        box-shadow: 0 0 24px rgba(99,179,237,0.18);
    }}
    .triage-label {{
        font-size: 0.76rem;
        color: #cbd5e0;
        font-weight: 500;
    }}
    .triage-value {{
        font-size: 0.70rem;
        color: #64748b;
        margin-top: 3px;
    }}
    .triage-current-note {{
        margin-top: 0.9rem;
        padding: 0.75rem 0.95rem;
        border-radius: 12px;
        background: rgba(99,179,237,0.06);
        border: 1px solid rgba(99,179,237,0.12);
        color: #cbd5e0;
        font-size: 0.84rem;
        line-height: 1.5;
    }}
    </style>
    </head>
    <body>
    <div class="triage-chart-card">
      <div class="triage-chart-title">Clinical Severity Comparison</div>
      <div class="triage-scale">

        <div class="triage-bar-wrap">
          <div class="triage-bar-outer">
            <div class="triage-bar-fill" style="height:28%;background:linear-gradient(180deg,#34d399,#15803d);"></div>
          </div>
          <div class="triage-label">Routine</div>
          <div class="triage-value">Stable</div>
        </div>

        <div class="triage-bar-wrap">
          <div class="triage-bar-outer">
            <div class="triage-bar-fill" style="height:62%;background:linear-gradient(180deg,#fbbf24,#d97706);"></div>
          </div>
          <div class="triage-label">Soon</div>
          <div class="triage-value">Follow-up</div>
        </div>

        <div class="triage-bar-wrap">
          <div class="triage-bar-outer">
            <div class="triage-bar-fill" style="height:96%;background:linear-gradient(180deg,#fb7185,#dc2626);"></div>
          </div>
          <div class="triage-label">Urgent</div>
          <div class="triage-value">Priority</div>
        </div>

        <div class="triage-bar-wrap">
          <div class="triage-bar-outer">
            <div class="triage-bar-fill" style="height:{current_height}%;background:{current_color};"></div>
          </div>
          <div class="triage-label">Patient Now</div>
          <div class="triage-value">{triage.title()}</div>
        </div>

        <div class="triage-bar-wrap">
          <div class="triage-bar-outer">
            <div class="triage-bar-fill" style="height:{conf_height}%;background:{conf_color};"></div>
          </div>
          <div class="triage-label">AI Confidence</div>
          <div class="triage-value">{confidence.title()}</div>
        </div>

      </div>
      <div class="triage-current-note">
        Current patient condition is classified as <strong>{triage}</strong> with
        <strong>{confidence}</strong> confidence based on the detected lab abnormalities.
      </div>
    </div>
    </body>
    </html>
    """
    components.html(html, height=300, scrolling=False)


def run_pipeline(report: str, question: str, category: str, progress_placeholder):
    steps = [
        ("Normalizing medical terms",            "Layer 1 — Medical Term Normalizer"),
        ("Rewriting patient question",           "Layer 1 — Patient-Friendly Query Rewriter"),
        ("Retrieving relevant report sections",  "Layer 2 — Hybrid Retriever (RAG)"),
        ("Generating specific answer",           "Layer 3 — LLM Generator"),
        ("Detecting abnormal values",            "Layer 4 — Abnormal Flag Detector"),
        ("Running content safety filter",        "Layer 5 — Content Filter"),
        ("Applying triage classification",       "Layer 5 — Red-Flag Detection"),
    ]

    def render(current):
        html = ""
        for i, (label, _) in enumerate(steps):
            if i < current:
                dot, color = "done", "#4b5563"
            elif i == current:
                dot, color = "active", "#93c5fd"
            else:
                dot, color = "waiting", "#374151"
            html += '<div class="step-row"><div class="step-dot ' + dot + '"></div><span style="color:' + color + '">' + label + '</span></div>'
        progress_placeholder.markdown('<div class="card">' + html + '</div>', unsafe_allow_html=True)

    render(0); normalized = normalize_terms(report)
    render(1); improved_q = rewrite_question(normalized, category, question)
    render(2)
    retrieved = hybrid_retrieve(
        st.session_state.get("chunks", []),
        st.session_state.get("embeddings", []),
        improved_q, top_k=5
    )
    render(3); explanation = generate_explanation(retrieved, improved_q, category)
    render(4); flags       = detect_flags(retrieved)
    render(5); explanation = content_filter(explanation)
    render(6); triage      = red_flag_triage(flags)

    time.sleep(0.3)
    progress_placeholder.empty()
    return explanation, flags, triage, retrieved


# ============================================================
#  PDF REPORT GENERATOR  (uses reportlab — always produces valid PDFs)
#  Install: pip install reportlab
# ============================================================
def generate_pdf_report(question, explanation, flags, triage, category, history):
    """Generate a PDF report.
    Uses reportlab when available and falls back to a pure-Python PDF writer when it is not.
    """

    def _clean_plain(text: str) -> str:
        text = str(text or "")
        text = re.sub(r'\*{1,3}', '', text)
        text = re.sub(r'#{1,6}\s*', '', text)
        text = text.replace('\r', '')
        text = re.sub(r'\n{3,}', '\n\n', text)
        replacements = {
            '—': '-', '–': '-', '“': '"', '”': '"', '’': "'", '•': '-',
            '≤': '<=', '≥': '>=', '→': '->', '\u00a0': ' '
        }
        for a, b in replacements.items():
            text = text.replace(a, b)
        return text.strip()

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

        abnormals = flags.get("abnormal_values", [])
        confidence = flags.get("confidence", "MEDIUM")
        severity = flags.get("severity", "LOW")
        urgent = flags.get("needs_doctor_urgently", False)
        now_str = datetime.datetime.now().strftime("%d %B %Y, %I:%M %p")

        C_BLUE = colors.HexColor("#63b3ed")
        C_NAVY = colors.HexColor("#1e3a8a")
        C_BODY = colors.HexColor("#1e293b")
        C_MUTED = colors.HexColor("#64748b")
        C_RED = colors.HexColor("#dc2626")
        C_AMBER = colors.HexColor("#d97706")
        C_GREEN = colors.HexColor("#15803d")
        C_LBG = colors.HexColor("#f1f5f9")
        TRIAGE_COLOR = {"URGENT": C_RED, "SOON": C_AMBER, "ROUTINE": C_GREEN}

        base = getSampleStyleSheet()

        def sty(name, parent="Normal", **kw):
            return ParagraphStyle(name, parent=base[parent], **kw)

        s_title = sty("s_title", "Title", fontSize=20, textColor=C_BLUE, spaceAfter=2, fontName="Helvetica-Bold")
        s_sub = sty("s_sub", "Normal", fontSize=8, textColor=C_MUTED, spaceAfter=6)
        s_meta = sty("s_meta", "Normal", fontSize=9, textColor=C_MUTED, spaceAfter=8)
        s_section = sty("s_sect", "Normal", fontSize=12, textColor=C_NAVY, fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4)
        s_body = sty("s_body", "Normal", fontSize=10, textColor=C_BODY, leading=15, spaceAfter=4)
        s_italic = sty("s_ital", "Normal", fontSize=10, textColor=C_BODY, fontName="Helvetica-Oblique", leading=14, spaceAfter=4)
        s_small = sty("s_small", "Normal", fontSize=8, textColor=C_MUTED, leading=12, spaceAfter=2)
        s_flag = sty("s_flag", "Normal", fontSize=10, textColor=C_RED, fontName="Helvetica-Bold", spaceAfter=2)
        s_disc = sty("s_disc", "Normal", fontSize=8, textColor=C_MUTED, fontName="Helvetica-Oblique", leading=12, spaceBefore=10)

        def clean_xml(text: str) -> str:
            text = _clean_plain(text)
            return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
            title="MediScan AI Report",
            author="MediScan AI",
        )
        story = []

        story.append(Paragraph("MediScan AI", s_title))
        story.append(Paragraph("Lab Report Analysis &mdash; Educational Use Only", s_sub))
        story.append(HRFlowable(width="100%", thickness=1, color=C_BLUE, spaceAfter=6))
        story.append(Paragraph(f"Generated: {now_str} &nbsp;&nbsp;|&nbsp;&nbsp; Panel: {clean_xml(category)}", s_meta))

        triage_label = {
            "URGENT": "URGENT - Immediate Medical Attention Required",
            "SOON": "SOON - Follow-Up Recommended",
            "ROUTINE": "ROUTINE - Values Within Acceptable Range",
        }.get(triage, "ROUTINE")
        banner_color = TRIAGE_COLOR.get(triage, C_GREEN)
        banner_tbl = Table([[Paragraph(f"<b>{triage_label}</b>", ParagraphStyle("ban", fontSize=11, textColor=colors.white, fontName="Helvetica-Bold"))]], colWidths=[170 * mm])
        banner_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), banner_color),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ]))
        story.append(Spacer(1, 4 * mm))
        story.append(banner_tbl)
        story.append(Spacer(1, 4 * mm))

        stats_data = [[
            Paragraph(f"<b>Confidence</b><br/>{confidence}", s_small),
            Paragraph(f"<b>Severity</b><br/>{severity}", s_small),
            Paragraph(f"<b>Urgent</b><br/>{'YES' if urgent else 'No'}", s_small),
        ]]
        stats_tbl = Table(stats_data, colWidths=[55 * mm, 55 * mm, 55 * mm])
        stats_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C_LBG),
            ("BOX", (0, 0), (-1, -1), 0.5, C_MUTED),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, C_MUTED),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(stats_tbl)
        story.append(Spacer(1, 5 * mm))

        if abnormals:
            story.append(HRFlowable(width="100%", thickness=0.5, color=C_MUTED, spaceAfter=4))
            story.append(Paragraph("Flagged Abnormal Values", s_section))
            for v in abnormals:
                story.append(Paragraph(f"&#x2022; {clean_xml(str(v))}", s_flag))
            story.append(Spacer(1, 3 * mm))

        story.append(HRFlowable(width="100%", thickness=0.5, color=C_MUTED, spaceAfter=4))
        story.append(Paragraph("Your Question", s_section))
        story.append(Paragraph(clean_xml(question), s_italic))
        story.append(Spacer(1, 3 * mm))

        story.append(Paragraph("AI Analysis", s_section))
        for para in clean_xml(explanation).split("\n\n"):
            para = para.strip()
            if para:
                story.append(Paragraph(para.replace("\n", "<br/>"), s_body))
        story.append(Spacer(1, 4 * mm))

        other = [h for h in history if h.get("question") != question]
        if other:
            story.append(HRFlowable(width="100%", thickness=0.5, color=C_MUTED, spaceAfter=4))
            story.append(Paragraph("Other Questions This Session", s_section))
            for item in other:
                icon = {"URGENT": "[URGENT]", "SOON": "[SOON]", "ROUTINE": "[OK]"}.get(item.get("triage"), "[OK]")
                story.append(Paragraph(f"<b>{icon}</b> {clean_xml(item.get('question', ''))}", s_body))
                preview = clean_xml(item.get("explanation", "")[:300])
                story.append(Paragraph(preview + "...", s_small))
                story.append(Spacer(1, 2 * mm))

        story.append(HRFlowable(width="100%", thickness=0.5, color=C_MUTED, spaceAfter=4))
        story.append(Paragraph(
            "DISCLAIMER: This report is generated by MediScan AI for educational purposes only and does not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider before making any health decisions. In case of emergency, contact local emergency services.",
            s_disc,
        ))

        doc.build(story)
        return buf.getvalue()

    except ModuleNotFoundError:
        pass

    abnormals = flags.get("abnormal_values", [])
    confidence = flags.get("confidence", "MEDIUM")
    severity = flags.get("severity", "LOW")
    urgent = flags.get("needs_doctor_urgently", False)
    now_str = datetime.datetime.now().strftime("%d %B %Y, %I:%M %p")

    lines = [
        "MediScan AI",
        "Lab Report Analysis - Educational Use Only",
        "",
        f"Generated: {now_str}",
        f"Panel: {_clean_plain(category)}",
        f"Triage: {triage}",
        f"Confidence: {confidence}",
        f"Severity: {severity}",
        f"Urgent: {'YES' if urgent else 'No'}",
        "",
    ]

    if abnormals:
        lines.append("Flagged Abnormal Values:")
        lines.extend([f"- {_clean_plain(v)}" for v in abnormals])
        lines.append("")

    lines.extend([
        "Your Question:",
        _clean_plain(question),
        "",
        "AI Analysis:",
        _clean_plain(explanation),
        "",
    ])

    other = [h for h in history if h.get("question") != question]
    if other:
        lines.append("Other Questions This Session:")
        for item in other:
            lines.append(f"- {_clean_plain(item.get('question', ''))}")
            lines.append(f"  {_clean_plain(item.get('explanation', '')[:250])}...")
        lines.append("")

    lines.append("DISCLAIMER: This report is generated by MediScan AI for educational purposes only and does not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider before making any health decisions.")

    text = "\n".join(lines)

    def _wrap(text_line: str, max_chars: int = 95):
        if not text_line:
            return [""]
        words = text_line.split()
        out, cur = [], ""
        for word in words:
            test = word if not cur else cur + " " + word
            if len(test) <= max_chars:
                cur = test
            else:
                if cur:
                    out.append(cur)
                cur = word
        if cur:
            out.append(cur)
        return out or [""]

    wrapped_lines = []
    for line in text.split("\n"):
        wrapped_lines.extend(_wrap(line))

    page_line_capacity = 48
    pages = [wrapped_lines[i:i + page_line_capacity] for i in range(0, len(wrapped_lines), page_line_capacity)] or [[""]]

    def _pdf_escape(s: str) -> str:
        return s.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')

    objects = []

    def add_obj(data: bytes):
        objects.append(data)
        return len(objects)

    font_obj = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_obj_ids = []
    content_obj_ids = []
    pages_obj_placeholder = add_obj(b"")

    width, height = 595, 842
    margin_left, margin_top = 50, 50
    line_height = 15

    for page in pages:
        commands = ["BT", "/F1 11 Tf", f"1 0 0 1 {margin_left} {height - margin_top} Tm", f"{line_height} TL"]
        for idx, line in enumerate(page):
            if idx > 0:
                commands.append("T*")
            commands.append(f"({_pdf_escape(line)}) Tj")
        commands.append("ET")
        stream = "\n".join(commands).encode("latin-1", errors="replace")
        content_id = add_obj(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
        content_obj_ids.append(content_id)
        page_obj_ids.append(add_obj(b""))

    kids = " ".join(f"{pid} 0 R" for pid in page_obj_ids).encode()
    objects[pages_obj_placeholder - 1] = b"<< /Type /Pages /Kids [ " + kids + b" ] /Count " + str(len(page_obj_ids)).encode() + b" /MediaBox [0 0 595 842] >>"
    catalog_obj = add_obj(b"<< /Type /Catalog /Pages " + str(pages_obj_placeholder).encode() + b" 0 R >>")

    for i, page_id in enumerate(page_obj_ids):
        content_id = content_obj_ids[i]
        objects[page_id - 1] = (
            b"<< /Type /Page /Parent " + str(pages_obj_placeholder).encode() + b" 0 R "
            + b"/Resources << /Font << /F1 " + str(font_obj).encode() + b" 0 R >> >> "
            + b"/Contents " + str(content_id).encode() + b" 0 R >>"
        )

    pdf = io.BytesIO()
    pdf.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(pdf.tell())
        pdf.write(f"{i} 0 obj\n".encode())
        pdf.write(obj)
        pdf.write(b"\nendobj\n")
    xref_pos = pdf.tell()
    pdf.write(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        pdf.write(f"{off:010d} 00000 n \n".encode())
    pdf.write(f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_obj} 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode())
    return pdf.getvalue()


# ============================================================
#  SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='padding:1.2rem 0 0.5rem'>
      <div style='font-family:"DM Serif Display",serif;font-size:1.4rem;color:#63b3ed;margin-bottom:4px'>MediScan AI</div>
      <div style='font-size:0.78rem;color:#4b5563;letter-spacing:0.5px'>GPT-4o + RAG Architecture</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div style="font-size:0.72rem;letter-spacing:1.6px;text-transform:uppercase;color:#63b3ed;margin-bottom:0.6rem">Load Report</div>', unsafe_allow_html=True)
    input_method = st.radio("", ["📎  Upload File", "📋  Paste Text"], label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)

    if input_method == "📎  Upload File":
        uploaded = st.file_uploader("Drop your lab report", type=["pdf","png","jpg","jpeg","tiff","bmp","webp"], label_visibility="collapsed")
        if uploaded and st.button("Load Report", use_container_width=True):
            with st.spinner("Extracting text..."):
                text = extract_from_upload(uploaded)
            if text.strip():
                with st.spinner("Building retrieval index..."):
                    chunks, embeddings = build_index(text)
                st.session_state.update({
                    "lab_report": text, "chunks": chunks, "embeddings": embeddings,
                    "report_loaded": True, "chat_history": []
                })
                with st.spinner("Classifying panel..."):
                    st.session_state.test_category = classify_category(text)
                st.success("Report loaded  " + str(len(chunks)) + " chunks indexed")
            else:
                st.error("Could not extract text. Try a clearer image or text-based PDF.")
    else:
        pasted = st.text_area("Paste your lab report here", height=220, label_visibility="collapsed", placeholder="Paste lab report text here...")
        if st.button("Load Report", use_container_width=True):
            if pasted.strip():
                with st.spinner("Building retrieval index..."):
                    chunks, embeddings = build_index(pasted.strip())
                st.session_state.update({
                    "lab_report": pasted.strip(), "chunks": chunks, "embeddings": embeddings,
                    "report_loaded": True, "chat_history": []
                })
                with st.spinner("Classifying panel..."):
                    st.session_state.test_category = classify_category(pasted)
                st.success("Report loaded  " + str(len(chunks)) + " chunks indexed")
            else:
                st.warning("Please paste some text first.")

    if st.session_state.report_loaded:
        st.markdown("---")
        st.markdown('<div style="font-size:0.72rem;letter-spacing:1.6px;text-transform:uppercase;color:#63b3ed;margin-bottom:0.6rem">Report Info</div>', unsafe_allow_html=True)
        st.markdown('<div class="cat-pill">🔬 ' + st.session_state.test_category + '</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.8rem;color:#4b5563;margin-top:8px">' + str(len(st.session_state.lab_report)) + ' chars · ' + str(len(st.session_state.chunks)) + ' chunks</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.72rem;letter-spacing:1.6px;text-transform:uppercase;color:#63b3ed;margin-bottom:0.6rem">Session Stats</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="metric-box"><div class="metric-val">' + str(st.session_state.total_questions) + '</div><div class="metric-lbl">Questions</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="metric-box"><div class="metric-val" style="color:#f87171">' + str(st.session_state.flagged_count) + '</div><div class="metric-lbl">Flagged</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear & Reset", use_container_width=True):
            for k in ["lab_report","report_loaded","chat_history","test_category","total_questions","flagged_count","chunks","embeddings"]:
                del st.session_state[k]
            st.rerun()

    st.markdown("---")
    st.markdown('<div style="font-size:0.75rem;color:#374151;line-height:1.6">For educational use only.<br>Not a substitute for medical advice.</div>', unsafe_allow_html=True)


# ============================================================
#  MAIN AREA
# ============================================================
st.markdown("""
<div class="hero-wrap">
  <div class="hero-icon">🩺</div>
  <div>
    <div class="hero-title">MediScan AI <span class="rag-badge">RAG</span></div>
    <div class="hero-sub">Upload your lab report — each question retrieves only the relevant sections and gives a specific, targeted answer</div>
  </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.report_loaded:
    col1, col2, col3 = st.columns(3)
    for col, icon, title, desc in [
        (col1, "📄", "Upload PDF or Image", "Supports PDF, JPG, PNG — GPT-4o Vision extracts all text"),
        (col2, "🔍", "RAG Retrieval",       "Your question retrieves only the relevant sections"),
        (col3, "💬", "Specific Answers",    "Each question gets a unique targeted answer"),
    ]:
        with col:
            st.markdown('<div class="card" style="text-align:center;padding:2rem 1.5rem"><div style="font-size:2.2rem;margin-bottom:12px">' + icon + '</div><div style="font-size:1rem;font-weight:500;color:#e2e8f0;margin-bottom:8px">' + title + '</div><div style="font-size:0.85rem;color:#64748b;line-height:1.55">' + desc + '</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="disclaimer">MediScan AI is for educational purposes only. Always consult a qualified healthcare professional.</div>', unsafe_allow_html=True)
    st.stop()

# Q&A
st.markdown('<div class="card-title">Ask About Your Report</div>', unsafe_allow_html=True)
col_q, col_btn = st.columns([5, 1])
with col_q:
    question = st.text_input("", placeholder="e.g. What does my HbA1c mean?  Do I need urgent care?", label_visibility="collapsed", key="question_input")
with col_btn:
    ask_clicked = st.button("Analyse", use_container_width=True)

quick_questions = ["Are any values abnormal?", "What should I be worried about?", "What disease could this indicate?", "Do I need to see a doctor urgently?"]
q_cols = st.columns(len(quick_questions))
for i, (qcol, qq) in enumerate(zip(q_cols, quick_questions)):
    with qcol:
        if st.button(qq, key="quick_" + str(i)):
            question = qq
            ask_clicked = True

if ask_clicked and question.strip():
    progress_ph = st.empty()
    try:
        explanation, flags, triage, retrieved = run_pipeline(
            st.session_state.lab_report, question,
            st.session_state.test_category, progress_ph
        )
        st.session_state.total_questions += 1
        if flags.get("abnormal_values"):
            st.session_state.flagged_count += 1
        st.session_state.chat_history.append({
            "question": question, "explanation": explanation,
            "flags": flags, "triage": triage, "retrieved": retrieved,
        })
    except Exception as e:
        progress_ph.empty()
        st.error("Pipeline error: " + str(e))
        st.stop()

if st.session_state.chat_history:
    latest     = st.session_state.chat_history[-1]
    flags      = latest["flags"]
    triage     = latest["triage"]
    abnormals  = flags.get("abnormal_values", [])
    confidence = flags.get("confidence", "MEDIUM")

    st.markdown("---")
    badge_cls  = {"URGENT":"badge-urgent","SOON":"badge-soon","ROUTINE":"badge-routine"}.get(triage,"badge-routine")
    badge_icon = {"URGENT":"🔴","SOON":"🟡","ROUTINE":"🟢"}.get(triage,"🟢")
    conf_color = {"HIGH":"#34d399","MEDIUM":"#fbbf24","LOW":"#f87171"}.get(confidence,"#fbbf24")

    hcol1, hcol2, hcol3 = st.columns([2, 2, 3])
    with hcol1:
        st.markdown('<div style="margin-top:4px"><span class="badge ' + badge_cls + '">' + badge_icon + ' ' + triage + '</span></div>', unsafe_allow_html=True)
    with hcol2:
        st.markdown('<div style="font-size:0.82rem;color:#64748b;margin-top:6px">Confidence: <span style="color:' + conf_color + ';font-weight:500">' + confidence + '</span></div>', unsafe_allow_html=True)
    with hcol3:
        if abnormals:
            chips = "".join('<span class="flag-chip">&#9873; ' + v + '</span>' for v in abnormals)
            st.markdown('<div style="margin-top:2px">' + chips + '</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if triage == "URGENT":
        st.markdown('<div class="alert-urgent">⚠️ <strong>Urgent attention required.</strong><br>One or more values need prompt medical review. Contact your doctor or visit a clinic as soon as possible.</div>', unsafe_allow_html=True)
    elif triage == "SOON":
        st.markdown('<div class="alert-soon">📋 <strong>Follow-up recommended.</strong><br>Some values are outside the normal range. Schedule an appointment at your earliest convenience.</div>', unsafe_allow_html=True)
    if confidence == "LOW":
        st.markdown('<div class="alert-info">ℹ️ Confidence is low — the report may be incomplete. Please verify with your healthcare provider.</div>', unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.8rem;color:#4b5563;margin-bottom:6px">❓ <em style="color:#93c5fd">' + latest["question"] + '</em></div>', unsafe_allow_html=True)
    _expl_clean = re.sub(r'\n{3,}', '\n\n', latest["explanation"])
    _expl_html  = _expl_clean.replace('\n\n', '</p><p>').replace('\n', '<br>')
    st.markdown('<div class="result-box"><p>' + _expl_html + '</p></div>', unsafe_allow_html=True)
    render_triage_chart(triage, confidence)

    st.markdown("<br>", unsafe_allow_html=True)
    try:
        pdf_bytes = generate_pdf_report(
            latest["question"], latest["explanation"],
            latest["flags"], latest["triage"],
            st.session_state.test_category,
            st.session_state.chat_history
        )
        st.download_button(
            label="📥  Download Full Report as PDF",
            data=pdf_bytes,
            file_name="MediScan_AI_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    except Exception as pdf_err:
        st.warning("PDF error: " + str(pdf_err))

    if len(st.session_state.chat_history) > 1:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📂  Previous questions (" + str(len(st.session_state.chat_history)-1) + ")", expanded=False):
            for item in reversed(st.session_state.chat_history[:-1]):
                t_icon  = {"URGENT":"🔴","SOON":"🟡","ROUTINE":"🟢"}.get(item["triage"],"🟢")
                preview = item["explanation"][:120].replace("\n"," ") + "..."
                st.markdown('<div class="history-item"><div class="history-q">' + t_icon + ' ' + item["question"] + '</div><div class="history-preview">' + preview + '</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="disclaimer">This explanation is for educational purposes only and is <strong>not a medical diagnosis</strong>. Always consult your doctor before making any health decisions.</div>', unsafe_allow_html=True)