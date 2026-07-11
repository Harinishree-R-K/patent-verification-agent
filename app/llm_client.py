"""
Thin, swappable LLM client wrapper.

Why this exists: the agents never call an LLM SDK directly. They call
`call_json(system, user)` on this module. That means swapping providers
(Anthropic <-> OpenAI <-> Gemini) is a one-file change, not a rewrite of
five agents. This also makes agents trivially testable via monkeypatching
`call_json` — see tests/test_pipeline_dry_run.py.
"""
from __future__ import annotations
import json
import os
import re
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root and sets os.environ from it

PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").lower()
MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")


def _strip_json_fences(raw: str) -> str:
    return re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()


def _call_anthropic(system: str, user: str, max_tokens: int = 1000) -> str:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    resp = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    for block in resp.content:
        if block.type == "text":
            return block.text
    return ""


def _call_openai(system: str, user: str, max_tokens: int = 1000) -> str:
    from openai import OpenAI

    client = OpenAI()  # reads OPENAI_API_KEY from env
    resp = client.chat.completions.create(
        model=os.environ.get("LLM_MODEL", "gpt-4o"),
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""

def _call_gemini(system: str, user: str, max_tokens: int = 1000) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = client.models.generate_content(
        model=os.environ.get("LLM_MODEL", "gemini-flash-latest"),
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max(max_tokens, 2000),
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return resp.text or ""


def call_raw(system: str, user: str, max_tokens: int = 1000) -> str:
    """Call the configured provider and return raw text."""
    if PROVIDER == "anthropic":
        return _call_anthropic(system, user, max_tokens)
    elif PROVIDER == "openai":
        return _call_openai(system, user, max_tokens)
    elif PROVIDER == "gemini":
        return _call_gemini(system, user, max_tokens)
    raise ValueError(f"Unknown LLM_PROVIDER: {PROVIDER}")


def call_json(system: str, user: str, max_tokens: int = 1000) -> Dict[str, Any]:
    """
    Call the LLM with an instruction to return ONLY JSON, then parse it.
    Raises ValueError if the model didn't return valid JSON after cleanup.
    """
    full_system = (
        system
        + "\n\nCRITICAL: Respond with ONLY valid JSON. No preamble, no markdown "
        "code fences, no explanation outside the JSON object."
    )
    raw = call_raw(full_system, user, max_tokens)
    cleaned = _strip_json_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {raw!r}") from e
