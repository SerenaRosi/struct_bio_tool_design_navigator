import os
import json
import time
import requests
import streamlit as st

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_groq_key():
    # 1) Streamlit secrets (works on Streamlit Cloud; also works on HF for Streamlit apps)
    if "GROQ_API_KEY" in st.secrets:
        return st.secrets["GROQ_API_KEY"]
    # 2) Fallback to environment variable (works on HF Spaces “Secrets” too)
    return os.getenv("GROQ_API_KEY", "")


# ----------------------------
# Data model
# ----------------------------
@dataclass
class Rule:
    id: str
    title: str
    description: str
    imagine_if: str
    questions: List[str]
    keywords: List[str]

    def as_text(self) -> str:
        parts = [
            self.title,
            self.description,
            self.imagine_if,
            " ".join(self.questions or []),
            " ".join(self.keywords or []),
        ]
        return "\n".join([p for p in parts if p])


def load_rules(path: str) -> List[Rule]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Rule(**r) for r in raw]


def pick_relevant_rules(rules: List[Rule], user_context: str, k: int = 4) -> List[Tuple[Rule, float]]:
    corpus = [r.as_text() for r in rules] + [user_context]
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(corpus)

    rule_vecs = X[:-1]
    user_vec = X[-1]

    sims = cosine_similarity(rule_vecs, user_vec)
    scored = [(rules[i], float(sims[i][0])) for i in range(len(rules))]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


# ----------------------------
# Optional LLM backends
# ----------------------------
def llm_ollama(prompt: str, model: str = "llama3.2") -> str:
    """
    Requires Ollama running locally:
      - install Ollama
      - `ollama pull llama3.2`
      - `ollama serve` (usually auto)
    Uses Ollama chat endpoint.
    """
    url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful research software engineering assistant."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    # Ollama returns {"message":{"content":...}, ...} for /api/chat
    return data.get("message", {}).get("content", "").strip()


def llm_groq(prompt: str, model: str = "llama-3.1-70b-versatile") -> str:
    """
    Groq-compatible OpenAI Chat Completions style endpoint.
    Needs GROQ_API_KEY in environment.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY environment variable.")

    # Groq API base can change; this is the commonly documented style.
    base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful research software engineering assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    r = requests.post(base_url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def call_llm(provider: str, prompt: str, model: str) -> str:
    if provider == "None":
        return ""
    if provider == "Ollama (local, free)":
        return llm_ollama(prompt, model=model)
    if provider == "Groq (hosted free tier)":
        return llm_groq(prompt, model=model)
    raise ValueError("Unknown provider")


# ----------------------------
# App UI
# ----------------------------
st.set_page_config(page_title="Structural Bioinformatics Rule Navigator", layout="wide")

st.title("Rule Navigator — Structural Bioinformatics Software Design")
st.caption("A rule-aware assistant based on your 'Ten Simple Rules' framework.")

with st.sidebar:
    st.header("Setup")
    rules_path = st.text_input("Path to rules.json", value="rules.json")
    top_k = st.slider("How many rules to focus on", min_value=2, max_value=6, value=4, step=1)

    st.divider()
    st.header("Optional LLM")
    provider = st.selectbox("Provider", ["None", "Ollama (local, free)", "Groq (hosted free tier)"])
    if provider == "Ollama (local, free)":
        model = st.text_input("Ollama model", value="llama3.2")
        st.caption("Requires Ollama running at OLLAMA_URL (default localhost:11434).")
    elif provider == "Groq (hosted free tier)":
        model = st.text_input("Groq model", value="llama-3.1-70b-versatile")
        st.caption("Requires GROQ_API_KEY env var.")
    else:
        model = ""

# Load rules
try:
    rules = load_rules(rules_path)
except Exception as e:
    st.error(f"Could not load rules from {rules_path}: {e}")
    st.stop()

st.subheader("Describe your tool")
col1, col2 = st.columns(2)

with col1:
    tool_name = st.text_input("Tool name (optional)")
    purpose = st.text_area("What does your tool do? (2–6 sentences)", height=140)
    users = st.text_input("Intended users (wet-lab / comp bio / RSE / mixed)")
    interaction = st.text_input("Use mode (CLI / GUI / web / notebook / plugin)")

with col2:
    inputs_outputs = st.text_area("Inputs & outputs (formats, APIs, files)", height=140)
    scale = st.text_input("Typical scale (single / dozens / thousands / HPC)")
    ai = st.text_input("AI/ML involved? (none / prediction / design / scoring + model family)")

context = f"""
Tool: {tool_name}
Purpose: {purpose}
Users: {users}
I/O: {inputs_outputs}
Scale: {scale}
Use mode: {interaction}
AI: {ai}
""".strip()

route_btn = st.button("Select relevant rules", type="primary")

if "selected" not in st.session_state:
    st.session_state.selected = []
if "answers" not in st.session_state:
    st.session_state.answers = {}  # rule_id -> {question: answer}

if route_btn:
    if not purpose.strip():
        st.warning("Please fill at least the tool purpose/description.")
    else:
        st.session_state.selected = pick_relevant_rules(rules, context, k=top_k)
        st.session_state.answers = {}  # reset answers

if st.session_state.selected:
    st.divider()
    st.subheader("Selected rules")
    for r, score in st.session_state.selected:
        st.markdown(f"**{r.id}: {r.title}** — match score `{score:.2f}`")

    st.divider()
    st.subheader("Work through the rules")

    for r, _score in st.session_state.selected:
        with st.expander(f"{r.id} — {r.title}", expanded=True):
            st.markdown("**Rule summary**")
            st.write(r.description)

            st.markdown("**Imagine if**")
            st.info(r.imagine_if)

            if r.id not in st.session_state.answers:
                st.session_state.answers[r.id] = {}

            st.markdown("**Questions**")
            for q in r.questions:
                key = f"{r.id}::{q}"
                prev = st.session_state.answers[r.id].get(q, "")
                ans = st.text_area(q, value=prev, key=key, height=80, placeholder="Answer briefly (or write 'unknown').")
                st.session_state.answers[r.id][q] = ans

    st.divider()
    st.subheader("Generate outputs")

    colA, colB = st.columns([1, 1])
    with colA:
        gen_report = st.button("Generate report (JSON + Markdown)", type="primary")
    with colB:
        gen_llm = st.button("Generate prioritized plan with LLM", disabled=(provider == "None"))

    # Build base report
    report = {
        "tool": {
            "name": tool_name,
            "purpose": purpose,
            "users": users,
            "inputs_outputs": inputs_outputs,
            "scale": scale,
            "interaction": interaction,
            "ai": ai,
        },
        "selected_rules": [
            {"id": r.id, "title": r.title, "match_score": score} for r, score in st.session_state.selected
        ],
        "answers": st.session_state.answers,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    def render_markdown_report(rep: Dict[str, Any]) -> str:
        md = []
        md.append(f"# Rule Navigator Report\n")
        md.append(f"## Tool\n")
        for k, v in rep["tool"].items():
            if v:
                md.append(f"- **{k}**: {v}")
        md.append("\n## Selected rules\n")
        for rr in rep["selected_rules"]:
            md.append(f"- **{rr['id']}**: {rr['title']} (match {rr['match_score']:.2f})")
        md.append("\n## Answers by rule\n")
        for rid, qa in rep["answers"].items():
            md.append(f"### {rid}\n")
            for q, a in qa.items():
                md.append(f"- **Q:** {q}\n  - **A:** {a if a else '—'}")
            md.append("")
        md.append("\n## Checklist\n")
        for rr in rep["selected_rules"]:
            rid = rr["id"]
            md.append(f"### {rid}: {rr['title']}")
            for q in rep["answers"].get(rid, {}).keys():
                md.append(f"- [ ] {q}")
            md.append("")
        return "\n".join(md)

    if gen_report:
        md = render_markdown_report(report)
        st.success("Report generated.")
        st.download_button("Download report.json", data=json.dumps(report, indent=2, ensure_ascii=False), file_name="report.json")
        st.download_button("Download report.md", data=md, file_name="report.md")
        st.markdown("### Preview (Markdown)")
        st.markdown(md)

    if gen_llm and provider != "None":
        # Make a tight, rule-grounded prompt to avoid vague output.
        selected_rules_text = []
        for r, score in st.session_state.selected:
            selected_rules_text.append(
                f"{r.id}: {r.title}\n"
                f"Imagine-if: {r.imagine_if}\n"
                f"Questions: " + " | ".join(r.questions)
            )
        prompt = f"""
You are helping design a structural bioinformatics software tool using a fixed set of rules.

USER TOOL CONTEXT
{context}

SELECTED RULES (do not invent new rules)
{chr(10).join(selected_rules_text)}

USER ANSWERS (may contain unknowns)
{json.dumps(st.session_state.answers, indent=2, ensure_ascii=False)}

TASK
Produce:
1) A prioritized to-do list (max 12 items) with rationale tied to specific rule IDs (e.g., "R3: ...").
2) Top 5 risks of silent failure or misinterpretation (each mapped to a rule).
3) 3 concrete deliverables the developer should produce next (e.g., README section, test dataset, plugin stub), mapped to rules.
Keep it practical and specific. If info is missing, state what is missing and propose the smallest next step.
""".strip()

        try:
            with st.spinner("Calling LLM..."):
                out = call_llm(provider, prompt, model=model)
            st.markdown("### LLM Output")
            st.markdown(out if out else "_No output returned._")
        except Exception as e:
            st.error(f"LLM call failed: {e}")
