# 🏠 Mortgage Document Assistant (RAG)

Ask plain-English questions about mortgage documents (Loan Estimates, closing costs,
PMI, escrow...) and get short answers **with page citations**, generated only from the
PDFs you provide.

Built with Retrieval-Augmented Generation (RAG):

```
PDFs ──► clean + chunk (800 chars, 150 overlap) ──► embeddings (all-MiniLM-L6-v2)
                                                          │
Question ──► embed ──► cosine-similarity search (top-k) ◄─┘
                              │
                              ▼
        grounded prompt (numbered passages + rules) ──► Gemini LLM ──► answer + [citations]
```

**Guardrails**
- The LLM may answer only from retrieved passages and must cite them like `[1]`.
- If no passage is similar enough, the app refuses *without* calling the LLM
  ("I couldn't find that in the provided documents.").
- Low temperature (0.1) to reduce made-up answers; no personalised financial advice.

## Setup (Windows / Mac / Linux)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                 # Mac/Linux: cp .env.example .env
```
Open `.env` and paste a free Gemini API key from https://aistudio.google.com/app/apikey

Put 2–5 mortgage PDFs in `data/` (see `data/README.md` for free public ones).

## Run

```bash
python build_index.py                         # read PDFs, build the vector index
python ask.py "What is a Loan Estimate?"      # quick test in the terminal
streamlit run app.py                          # chat UI in the browser
pytest -q                                     # offline unit tests
```

## Evaluation

1. Edit `eval/questions.json`: write ~20 questions your PDFs can answer, with a keyword
   that appears in the correct passage. Keep a few out-of-scope questions.
2. Run:
```bash
python -m eval.evaluate          # Hit@1, Hit@3, MRR, retrieval latency
python -m eval.evaluate --llm    # + citation rate and out-of-scope refusal rate
```
Results are saved to `eval/results.json`.

**Next experiments:** try `--chunk-size 500 --overlap 100`
in `build_index.py`, a different `k`, or a stronger embedding model, and compare scores.

## Project structure
```
rag/ingest.py      PDF loading, text cleaning, overlapping chunking
rag/retriever.py   embeddings + vector index (save/load, cosine search)
rag/generator.py   grounded prompt, Gemini call, refusal guardrail
app.py             Streamlit chat UI with expandable source passages
ask.py             command-line Q&A
eval/evaluate.py   retrieval + answer-quality metrics
tests/             offline unit tests (fake embedder + fake LLM)
```
