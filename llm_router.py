"""
Gemma 4 question interpreter for InsightData.

Role of the model (and its limits):
  * It sees ONLY the dataset schema: column names, column types, the distinct values of
    categorical columns, and the date range. It never sees numeric cell values.
  * It turns the user's free-form question into one canonical question that the existing
    rule-based engine (analysis_engine.py) already understands, or says that the data
    cannot answer it / that a clarification is needed.
  * Every number in every answer is still computed by analysis_engine.py with Pandas.
    Model text is never used as a calculation result.

If the API key is missing or the API fails, the original question is passed straight to
the existing engine, so the app keeps working exactly as before.
"""
import json
import logging
import os
import re
import time

import pandas as pd

try:  # optional: load GEMINI_API_KEY from a local .env file
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

from analysis_engine import analyze_question, identify_columns

MODEL = os.environ.get("GEMMA_MODEL", "gemma-4-26b-a4b-it")
TIMEOUT_MS = 20000
MAX_VALUES_PER_COLUMN = 40
MAX_QUESTION_CHARS = 500

logger = logging.getLogger("insightdata.gemma")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [Gemma] %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False

_client = None


# --------------------------------------------------------------------------- schema
def build_schema_text(df: pd.DataFrame) -> str:
    """Describes the dataset WITHOUT exposing numeric values."""
    numeric_cols, categorical_cols, date_cols = identify_columns(df)
    lines = [f"Total rows: {len(df)}"]

    lines.append("Numeric columns (measures you can sum/average/compare):")
    lines += [f'  - "{c}"' for c in numeric_cols] or ["  (none)"]

    lines.append("Categorical columns (groupable / filterable), with their exact values:")
    if not categorical_cols:
        lines.append("  (none)")
    for c in categorical_cols:
        vals = [v for v in df[c].dropna().astype(str).str.strip().unique() if v]
        shown = vals[:MAX_VALUES_PER_COLUMN]
        extra = f" ... ({len(vals)} distinct values in total, list truncated)" if len(vals) > len(shown) else ""
        lines.append(f'  - "{c}": {json.dumps(shown, ensure_ascii=False)}{extra}')

    lines.append("Date columns:")
    if not date_cols:
        lines.append("  (none)")
    for c in date_cols:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce", format="mixed")
        except (TypeError, ValueError):
            parsed = pd.to_datetime(df[c], errors="coerce")
        parsed = parsed.dropna()
        span = f"{parsed.min().date()} to {parsed.max().date()}" if not parsed.empty else "no valid dates"
        lines.append(f'  - "{c}": {span}')
    return "\n".join(lines)


PROMPT_TEMPLATE = """You are the question-interpretation step of a data-analysis tool. You do NOT calculate anything and you never state numbers about the data. A separate engine does all calculations.

DATASET SCHEMA (the only information you have about the data):
{schema}

YOUR JOB: read the user's question and reply with ONE JSON object, nothing else (no markdown, no code fences):
{{"status": "ok" | "unanswerable" | "clarify", "rewritten_question": "...", "message": "..."}}

status "ok": the schema can answer the question. Put in "rewritten_question" ONE question in exactly one of these templates. Use the EXACT column names and EXACT category values from the schema, spelled as shown.
  - total:            What is the total <numeric column> [for <category value>]?
  - average:          What is the average <numeric column> [for <category value>]?
  - highest/lowest group: Which <categorical column> has the highest <numeric column>?   (or "lowest")
  - highest/lowest record: Show the highest <numeric column>.   (or "lowest")
  - spread:           What is the difference between the highest and lowest <numeric column>?
  - compare all groups: Compare <numeric column> across <categorical column>s.
  - compare two:      Compare <numeric column> between <category value A> and <category value B>.
  - trend:            How does <numeric column> change over time?
  - first vs last date: Compare <numeric column> between the earliest and latest dates.
  - record count:     How many records are in the dataset?   (or "How many records are there for <category value>?")
Time periods: if the user mentions one of "today", "yesterday", "this/last week", "this/last month", "this/last quarter", "this/last year", or "last N days/weeks/months", copy that phrase unchanged to the end of rewritten_question (for example "... last month?"). Do not convert it to dates.
Leave "message" empty for status ok.

status "unanswerable": the schema has no column that can answer it (for example the user asks about a measure that does not exist, or the question is not about this data). Put a short plain-language explanation in "message", naming the available columns. Leave "rewritten_question" empty.

status "clarify": the question could reasonably refer to two or more columns or categories and you cannot tell which. Put one short clarifying question in "message", naming the options. Leave "rewritten_question" empty.

Rules: never include any number about the data in your reply; never guess a column that is not in the schema; if you are unsure whether the data can answer it, prefer "clarify" over "ok".

USER QUESTION:
{question}
"""


# --------------------------------------------------------------------------- model call
def _get_client():
    """Returns (client, error). The key is read from the environment only."""
    global _client
    key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        return None, "GEMINI_API_KEY is not set"
    if _client is None:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            return None, "the google-genai package is not installed (pip install -r requirements.txt)"
        _client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=TIMEOUT_MS))
    return _client, None


def _call_model(prompt: str) -> str:
    """The single place that talks to the Gemini API. Raises on failure."""
    client, err = _get_client()
    if err:
        raise RuntimeError(err)
    from google.genai import types
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=1024),
    )
    return response.text or ""


def _describe_error(exc: Exception) -> str:
    """Short, user-safe description of an API failure (never includes the key)."""
    code = getattr(exc, "code", None)
    text = str(exc)
    if isinstance(exc, RuntimeError):
        return text
    if code in (401, 403) or (code == 400 and "api key" in text.lower()):
        return "the Gemini API rejected the API key (check GEMINI_API_KEY)"
    if code == 404:
        return f"model '{MODEL}' was not found or is not available to this key"
    if code == 429:
        return "the Gemini API rate limit or quota was reached"
    if isinstance(code, int) and code >= 500:
        return "the Gemini API is temporarily unavailable"
    if code:
        return f"the Gemini API returned an error (HTTP {code})"
    return "the Gemini API could not be reached (network error or timeout)"


def _parse_reply(text: str):
    """Extracts and validates the JSON object. Returns dict or None."""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start:end + 1])
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("status") not in ("ok", "unanswerable", "clarify"):
        return None
    rewritten = data.get("rewritten_question")
    message = data.get("message")
    rewritten = rewritten.strip() if isinstance(rewritten, str) else ""
    message = message.strip() if isinstance(message, str) else ""
    if data["status"] == "ok" and not (0 < len(rewritten) <= MAX_QUESTION_CHARS):
        return None
    if data["status"] != "ok" and not message:
        return None
    return {"status": data["status"], "rewritten_question": rewritten, "message": message}


def interpret_question(df: pd.DataFrame, question: str):
    """Returns (interpretation_or_None, info). info always has latency_ms and error."""
    info = {"error": None, "latency_ms": None}
    prompt = PROMPT_TEMPLATE.format(schema=build_schema_text(df), question=question.strip()[:MAX_QUESTION_CHARS * 2])
    started = time.time()
    try:
        raw = _call_model(prompt)
    except Exception as exc:  # network, quota, auth, missing key/package ...
        info["error"] = _describe_error(exc)
        logger.warning("call FAILED model=%s reason=%s | detail=%s: %s", MODEL, info["error"], type(exc).__name__, str(exc)[:200])
        return None, info
    info["latency_ms"] = int((time.time() - started) * 1000)
    parsed = _parse_reply(raw)
    if parsed is None:
        info["error"] = "the model reply could not be understood"
        logger.warning("call OK but reply unusable model=%s latency=%dms", MODEL, info["latency_ms"])
        return None, info
    logger.info("call OK model=%s latency=%dms status=%s rewritten=%r", MODEL, info["latency_ms"], parsed["status"], parsed["rewritten_question"])
    return parsed, info


# --------------------------------------------------------------------------- public entry
def analyze_with_llm(df: pd.DataFrame, question: str) -> dict:
    """
    Same return shape as analyze_question(), plus an "llm" field describing what Gemma did.
    All numbers come from analysis_engine.analyze_question().
    """
    meta = {"model": MODEL, "used": False, "status": None, "interpreted_question": None,
            "latency_ms": None, "error": None, "fallback": False}

    if df is None or df.empty:
        result = analyze_question(df, question)
        result["llm"] = meta
        return result

    interp, info = interpret_question(df, question)
    meta["latency_ms"] = info["latency_ms"]

    # Gemma unavailable or unusable -> existing behaviour, with an honest note on failures.
    if interp is None:
        meta["error"], meta["fallback"] = info["error"], True
        result = analyze_question(df, question)
        if not result.get("success") and isinstance(result.get("answer"), str):
            result["answer"] += f" (Note: the AI interpreter was unavailable - {info['error']} - so built-in matching rules were used.)"
        result["llm"] = meta
        return result

    meta["used"], meta["status"] = True, interp["status"]

    if interp["status"] == "unanswerable":
        return {"success": False, "question": question, "answer": interp["message"],
                "operation": "llm_unanswerable", "error": "The dataset cannot answer this question.", "llm": meta}

    if interp["status"] == "clarify":
        return {"success": False, "question": question, "answer": interp["message"],
                "operation": "clarification_needed", "error": "Clarification needed.", "llm": meta}

    # status ok: the engine computes everything from the rewritten, canonical question.
    meta["interpreted_question"] = interp["rewritten_question"]
    result = analyze_question(df, interp["rewritten_question"])
    if not result.get("success"):
        original = analyze_question(df, question)  # keep old behaviour if the rewrite lost something
        if original.get("success"):
            result, meta["fallback"] = original, True
    result["question"] = question
    result["llm"] = meta
    return result
