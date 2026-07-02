# Approach Document — SHL Assessment Recommender

## Architecture
Stateless FastAPI service. Two-stage LLM pipeline per turn: 
1. **SCOPE** (classify intent, extract requirements, detect off-topic/injection) 
2. **SELECT** (pick 1-10 from retrieved candidates) or **COMPARE** (grounded Q&A). 

Retrieval uses BM25 over catalog name+description fields, `top_k=30` candidates handed to the selection LLM. Grounding enforced structurally — model never invents names/URLs, output is matched against real catalog items; unmatched selections are dropped.

## Catalog
377 official SHL Individual Test Solutions, scraped from assignment-provided source (live shl.com catalog page was redirected/gated at time of build, confirmed via manual site check).

## Refinement handling
Initial version re-derived shortlist from scratch each turn, dropping prior items on "also add X" requests. Fixed by adding explicit persistence instruction to selection prompt (keep prior items unless user asks to remove) and requiring the reply text to explicitly name every chosen assessment, since the API's stateless Message schema only carries plain text — that reply is the only way prior recommendations survive into the next turn's history.

## Evaluation
Tested against 10 provided conversation traces using a Groq-simulated multi-turn user. Behavior probes (refuse off-topic, refuse prompt injection, no-recommend on vague turn 1, honor refinement) passed 4/4. Baseline mean Recall@10 measured at 34.6%. Debug tracing on worst-performing traces showed the primary bottleneck was retrieval, not selection: expected items (e.g. "Global Skills Assessment" for a sales-reskilling query) often ranked far outside the top-15 BM25 candidates due to weak lexical/semantic overlap. 

Applied fixes: increased `top_k` to 30, added query expansion via LLM-generated synonym phrases merged into the candidate pool, and added precise-match guidance to the selection prompt to avoid picking superficially similar but incorrect items. Offline spot-check confirmed previously-missing items became retrievable after these changes; full re-verification was blocked by Groq API daily quota exhaustion before final submission.

## What didn't work
An early version of the catalog included a fabricated placeholder test item that nearly went undetected; caught via manual sanity check comparing live API output against known sample data. Pure BM25 keyword retrieval underperforms on semantically-related but lexically-distant queries — a known limitation; embedding-based or hybrid retrieval would likely improve recall further given more time.

## AI tools used
Agentic coding tools (Antigravity, Codex) for implementation, debugging, and deployment; used for writing/editing `agent.py`, running diagnostic traces, and managing git/deploy workflow. All design decisions (retry logic, prompt structure, retrieval strategy, bug root-causing) were reviewed and directed manually before implementation.
