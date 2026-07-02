"""Thin wrapper around Groq's OpenAI-compatible chat completions API."""
import json
import os

from openai import OpenAI

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        _client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    return _client


def call_json(system: str, user: str, temperature: float = 0.0) -> dict:
    """Call the model and parse a JSON object response. Retries once on
    malformed JSON, which happens occasionally with smaller open models."""
    for attempt in range(2):
        resp = client().chat.completions.create(
            model=GROQ_MODEL,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        raw = resp.choices[0].message.content
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            if attempt == 0:
                continue
            raise
    raise RuntimeError("unreachable")
