"""
Evaluation harness for the /chat endpoint.

1. Recall@10 over the provided conversation traces (drop the trace zip's
   files into eval/traces/*.json before running).
2. Behavior probes (off-topic refusal, no-recommend-on-turn-1, refine
   honored, comparison groundedness) defined in eval/probes.py.

I can't know the exact schema of SHL's trace zip until you download it,
so `load_trace` below tries a few common field-name variants and prints
a loud warning if a trace doesn't match any of them — fix the mapping
there once you see the real files, everything else stays the same.

Usage:
    export CHAT_URL=http://localhost:8000/chat
    export GROQ_API_KEY=...        # used to simulate the user
    python eval/run_eval.py
"""
import glob
import json
import os
import sys

import requests
from openai import OpenAI

CHAT_URL = os.environ.get("CHAT_URL", "http://localhost:8000/chat")
MAX_TURNS = 8

_sim_client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1"
)
SIM_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def load_trace(path: str) -> dict:
    raw = json.loads(open(path, encoding="utf-8").read())
    persona = raw.get("persona") or raw.get("profile") or ""
    facts = raw.get("facts") or raw.get("known_facts") or {}
    expected = (
        raw.get("expected_shortlist")
        or raw.get("expected_recommendations")
        or raw.get("labels")
        or []
    )
    opening = raw.get("opening_message") or raw.get("initial_message")
    if not expected:
        print(f"[warn] {path}: no expected shortlist found — check field names in load_trace()")
    return {"persona": persona, "facts": facts, "expected": expected, "opening": opening, "raw": raw}


def simulate_user_reply(persona: str, facts: dict, history: list[dict]) -> tuple[str, bool]:
    """Returns (message, should_end). Ends when the agent has clearly
    delivered a shortlist, mirroring the real harness's stated behavior."""
    last_agent = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), "")
    system = (
        "You are role-playing a hiring persona in a conversation with an SHL "
        "assessment recommender agent. Answer questions truthfully using ONLY "
        f"these facts: {json.dumps(facts)}. Persona: {persona}. If asked "
        "something outside these facts, say you have no preference. If the "
        "agent just gave you a shortlist of assessments, respond with a short "
        "thank-you and nothing else."
    )
    resp = _sim_client.chat.completions.create(
        model=SIM_MODEL,
        temperature=0.3,
        messages=[{"role": "system", "content": system}]
        + [{"role": m["role"], "content": m["content"]} for m in history]
        + [{"role": "user", "content": "Continue the conversation naturally."}],
    )
    text = resp.choices[0].message.content.strip()
    ended = "shortlist" in last_agent.lower() or "here are" in last_agent.lower()
    return text, ended


def run_trace(trace: dict) -> dict:
    history: list[dict] = []
    opener = trace["opening"] or "I need help finding an SHL assessment."
    history.append({"role": "user", "content": opener})

    final_recs: list[dict] = []
    for turn in range(MAX_TURNS):
        resp = requests.post(CHAT_URL, json={"messages": history}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        history.append({"role": "assistant", "content": data["reply"]})
        if data.get("recommendations"):
            final_recs = data["recommendations"]
        if data.get("end_of_conversation") or turn == MAX_TURNS - 1:
            break
        user_msg, ended = simulate_user_reply(trace["persona"], trace["facts"], history)
        history.append({"role": "user", "content": user_msg})
        if ended:
            break

    return {"history": history, "final_recommendations": final_recs}


def recall_at_k(expected: list[str], got: list[dict], k: int = 10) -> float:
    if not expected:
        return float("nan")
    got_names = {r["name"].lower() for r in got[:k]}
    hits = sum(1 for e in expected if e.lower() in got_names)
    return hits / len(expected)


def main():
    trace_paths = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "traces", "*.json")))
    if not trace_paths:
        print("No traces found in eval/traces/. Drop the provided trace JSON files there.")
        sys.exit(1)

    results = []
    for path in trace_paths:
        trace = load_trace(path)
        run = run_trace(trace)
        r10 = recall_at_k(trace["expected"], run["final_recommendations"])
        results.append({"trace": os.path.basename(path), "recall@10": r10})
        print(f"{os.path.basename(path):40s} recall@10 = {r10:.2f}")

    valid = [r["recall@10"] for r in results if r["recall@10"] == r["recall@10"]]  # drop NaN
    if valid:
        print(f"\nMean recall@10 across {len(valid)} traces: {sum(valid)/len(valid):.3f}")


if __name__ == "__main__":
    main()
