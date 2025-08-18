import os
import json
import re
import pandas as pd
import streamlit as st
from typing import List, Dict

# ---- LangChain + Gemini ----
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ---------- CONFIG ----------
MODEL_NAME = "gemini-2.0-flash"  # per request
EMBED_MODEL = "models/embedding-001"

st.set_page_config(page_title="RAG Service Quality & Sentiment", layout="wide")
st.title("Insurance Agents : SQA")

api_key = st.text_input("Enter your Google Gemini API key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# ------------- HELPERS -------------
def load_json_files(files: List[str]) -> List[Dict]:
    data = []
    for fp in files:
        try:
            with open(fp, "r") as f:
                obj = json.load(f)
                if isinstance(obj, list):
                    data.extend(obj)
                else:
                    data.append(obj)
        except Exception as e:
            st.error(f"Failed to load {fp}: {e}")
    return data

def chunk_docs(docs: List, key_map: Dict[str, str]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    texts = []
    for d in docs:
        if isinstance(d, dict):
            parts = []
            for k, v in d.items():
                if isinstance(v, (dict, list)):
                    parts.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
                else:
                    parts.append(f"{k}: {v}")
            full = "\n".join(parts)
        else:
            # fallback: just convert the raw object to string
            full = str(d)
        for c in text_splitter.split_text(full):
            texts.append(c)
    return texts


def build_vectorstore(texts):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    vs = FAISS.from_texts(texts, embedding=embeddings)
    return vs

def run_llm(grounding: str, transcript: str):
    template = """
You are an auditing assistant evaluating service calls. Use the *grounding* facts from policy/contracts and then analyze the *transcript*.

Return a JSON object with EXACT keys:
- behavior_tone_score_1_to_10: integer 1-10 (10 = excellent empathy & clarity)
- resolution_y_n: "y" or "n" (was the customer's query resolved or a clear next step provided?)
- profanity_y_n: "y" or "n" (any profanity used by agent or customer?)
- summary_<=20_words: string (<=20 words, key details only)
- topic_<=20_words: string (<=20 words)

Be concise, deterministic, and rely on grounding for benefits/coverage questions.

GROUNDING:
{grounding}

TRANSCRIPT:
{transcript}
"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0) | JsonOutputParser()
    return chain.invoke({"grounding": grounding, "transcript": transcript})

def analyze(transcripts: List[Dict], vectorstore) -> pd.DataFrame:
    results = []
    for t in transcripts:
        info = t.get("info", {})
        transcript_text = info.get("transcript", "")
        # retrieve grounding
        retriever = vectorstore.as_retriever(k=4)
        docs = retriever.get_relevant_documents(transcript_text)
        grounding = "\n\n".join(d.page_content for d in docs)

        try:
            out = run_llm(grounding, transcript_text)
        except Exception as e:
            # fallback minimal heuristic if LLM fails
            def _short(s, n=20):
                w = s.replace("\n"," ").split()
                return " ".join(w[:n])
            out = {
                "behavior_tone_score_1_to_10": 7,
                "resolution_y_n": "y" if any(w in transcript_text.lower() for w in ["submit","covered","you can","we cover","fill"]) else "n",
                "profanity_y_n": "y" if any(p in transcript_text.lower() for p in ["damn","shit","fuck","hell"]) else "n",
                "summary_<=20_words": _short(re.sub(r"(Customer:|Agent:|Provider:)\s*","", transcript_text)),
                "topic_<=20_words": _short(transcript_text)
            }

        results.append({
            "transcript_id": t.get("transcript_id"),
            "contract_id": t.get("contract_id"),
            "callDate": info.get("callDate"),
            "agents": info.get("agentsInvolved"),
            **out
        })
    return pd.DataFrame(results)

# ------------- SIDEBAR: DATA -------------
st.sidebar.header("Data Sources")
uploaded_contracts = st.sidebar.file_uploader("Add Contract Policy JSON files", type=["json"], accept_multiple_files=True)
uploaded_transcripts = st.sidebar.file_uploader("Add Transcripts JSON files", type=["json"], accept_multiple_files=True)

# Seed with examples if available in ./data or current directory
default_contract_paths = [p for p in ["Contract_12345.json"] if os.path.exists(p)]
default_transcript_paths = [p for p in ["transcripts.json"] if os.path.exists(p)]

contracts = []
transcripts = []

if default_contract_paths:
    contracts.extend(load_json_files(default_contract_paths))
if default_transcript_paths:
    transcripts.extend(load_json_files(default_transcript_paths))

if uploaded_contracts:
    for uf in uploaded_contracts:
        contracts.extend(json.load(uf))
if uploaded_transcripts:
    for uf in uploaded_transcripts:
        obj = json.load(uf)
        if isinstance(obj, list):
            transcripts.extend(obj)
        else:
            transcripts.append(obj)

st.sidebar.write(f"Contracts loaded: {len(contracts)}")
st.sidebar.write(f"Transcripts loaded: {len(transcripts)}")

# ------------- BUILD RAG INDEX -------------
if st.button("Build / Rebuild Index"):
    with st.spinner("Building vector index from contracts & transcripts..."):
        texts = []
        if contracts:
            texts += chunk_docs(contracts, {})
        # Optionally include transcripts as weak grounding (metadata/ids)
        if transcripts:
            texts += chunk_docs(transcripts, {})
        if not texts:
            st.warning("No documents to index. Please upload data.")
        else:
            vs = build_vectorstore(texts)
            st.session_state["vs"] = vs
            st.success(f"Indexed {len(texts)} chunks.")

# ------------- RUN ANALYSIS -------------
vs = st.session_state.get("vs")
if vs and transcripts:
    if st.button("Run Service Quality Analysis"):
        with st.spinner("Scoring calls via LLM..."):
            df = analyze(transcripts, vs)
            st.session_state["results_df"] = df
            st.success("Analysis complete.")

# ------------- RESULTS & DASHBOARD -------------
df = st.session_state.get("results_df")
if df is not None and not df.empty:
    st.subheader("Results Table")
    st.dataframe(df, use_container_width=True)

    # Save CSV
    out_csv = "service_quality_analysis.csv"
    df.to_csv(out_csv, index=False)
    st.download_button("Download CSV", data=df.to_csv(index=False), file_name=out_csv, mime="text/csv")

    st.subheader("Dashboard")
    col1, col2, col3 = st.columns(3)
    with col1:
        pos = (df["resolution_y_n"]=="y").sum()
        neg = (df["resolution_y_n"]=="n").sum()
        st.metric("Resolved Calls (y)", pos)
        st.metric("Unresolved Calls (n)", neg)
    with col2:
        pos_sent = (df.get("sentiment","positive")=="positive").sum() if "sentiment" in df else None
        neg_sent = (df.get("sentiment","negative")=="negative").sum() if "sentiment" in df else None
        if pos_sent is not None:
            st.metric("Positive sentiment (approx.)", pos_sent)
            st.metric("Negative sentiment (approx.)", neg_sent)
    with col3:
        st.write("Areas to Focus (Resolution = n)")
        st.dataframe(df[df["resolution_y_n"]=="n"][["transcript_id","topic_<=20_words","summary_<=20_words"]], use_container_width=True)

    st.caption("Note: Sentiment column can be added by extending the prompt or computing polarity from model output.")

else:
    st.info("Upload data and build the index, then run the analysis.")

st.sidebar.caption("Powered by LangChain + Gemini 2.0 Flash + Streamlit")