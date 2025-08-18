# RAG-based Service Quality & Sentiment Analyzer

This app uses **LangChain**, **Gemini 2.0 Flash**, and **Streamlit** to analyze customer support transcripts with contract/policy grounding.

## Features
- RAG index over contracts (+ transcripts) with FAISS.
- LLM-based scoring:
  - behavior_tone_score_1_to_10
  - resolution_y_n
  - profanity_y_n
  - summary_<=20_words
  - topic_<=20_words
- Dashboard: sentiment/resolution counts and "Areas to Focus" (Resolution = n).
- Upload more JSON files and re-run.
- Export CSV.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY=YOUR_KEY
streamlit run app.py
```
Place your data files next to `app.py` (e.g., `transcripts.json`, `Contract_12345.json`) or upload via the sidebar.

## Notes
- The Gemini model name is set to `gemini-2.0-flash`.
- Embeddings: `models/embedding-001` from the Gemini API.
- For on-prem or other providers, swap `ChatGoogleGenerativeAI` and the embeddings.