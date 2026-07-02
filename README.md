# SHL Assessment Recommender

Conversational agent that turns a vague hiring need into a grounded
shortlist of SHL **Individual Test Solutions**, via `POST /chat`.

## Architecture

```
scraper/scrape_shl.py   -> data/catalog.json   (Individual Test Solutions only)
app/retrieval.py         BM25 search over the catalog (rank_bm25)
app/llm.py                Groq (OpenAI-compatible) chat completions, JSON mode
app/agent.py              orchestration: scope-check -> clarify/recommend/refine/compare
app/main.py                FastAPI: GET /health, POST /chat
eval/run_eval.py          recall@10 against provided traces (simulated user via Groq)
eval/probes.py            behavior probes: refusal, no-recommend-turn-1, refinement
```

**Why BM25, not embeddings.** The catalog is a few hundred short,
keyword-dense records. BM25 is deterministic, needs no embeddings API
(Groq doesn't serve one), and avoids shipping torch/sentence-transformers
into a free-tier container with cold starts. Tradeoff: it's weaker on
pure semantic paraphrase ("someone who talks to clients a lot" vs.
"stakeholder management") — mitigated by having the LLM first rewrite the
conversation into a dense `requirements_summary` before it hits BM25, so
the query itself is already keyword-rich.

**Why grounding is structural, not just prompted.** The LLM is only ever
allowed to *choose* from a candidate list we retrieved from the real
catalog; the final `recommendations` array is built in Python by looking
up the model's chosen names in that candidate list. If a chosen name
doesn't match anything real, it's silently dropped. This is what
guarantees "every URL comes from the scraped catalog" rather than relying
on the model not hallucinating.

**Why refine "just works" statelessly.** Every turn re-reads the *entire*
message history and re-extracts a fresh `requirements_summary` — there's
no separate mutable state to keep in sync. "Actually, add personality
tests" is just one more sentence the extractor sees.

**Turn-cap handling.** If we're 3+ user turns in and still missing info,
the agent stops asking and commits to a best-effort shortlist from
whatever it has, rather than risking the 8-turn cap on endless
clarification.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY
```

### 1. Scrape the catalog (run locally — shl.com isn't reachable from
   sandboxed dev environments without general internet access)

```bash
cd scraper
python scrape_shl.py --out ../data/catalog.json
```

The scraper's CSS selectors are marked `# ADJUST ME` — SHL's exact markup
wasn't inspectable from this environment, so treat it as a strong
starting point and adjust selectors against the real page once you load
it in a browser. A 6-item `data/catalog.sample.json` is included so you
can run the whole stack before the real scrape is ready:
`cp data/catalog.sample.json data/catalog.json`.

### 2. Run the API

```bash
export GROQ_API_KEY=...
uvicorn app.main:app --reload
curl localhost:8000/health
```

### 3. Evaluate

```bash
# drop the provided trace files into eval/traces/*.json first
export CHAT_URL=http://localhost:8000/chat
python eval/run_eval.py     # recall@10
python eval/probes.py       # behavior probes
```

`eval/run_eval.py` tries a few common field names for the trace schema
(`persona`/`profile`, `facts`/`known_facts`, `expected_shortlist`/
`expected_recommendations`/`labels`) and prints a warning if none match —
adjust `load_trace()` once you see the real trace JSON.

## Deployment

Both a `Procfile` (Render/Railway) and `Dockerfile` (Fly/HF Spaces/Railway)
are included — pick whichever platform you land on. Either way, set
`GROQ_API_KEY` and `CATALOG_PATH` as environment variables, and make sure
the built `data/catalog.json` is included in the deployed image (it's not
gitignored on purpose).

## Known limitations / what I'd do next with more time

- BM25 is weaker than embeddings on semantic paraphrase; would add a
  hybrid re-rank if recall@10 on the real traces comes in low.
- The scraper's selectors are unverified against live markup — first
  real run needs a manual spot-check pass.
- Comparison currently needs the two assessment names to fuzzy-match the
  catalog; a nickname/alias table would make it more robust to how users
  actually phrase product names.
