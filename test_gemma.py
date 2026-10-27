"""
Tests for the Gemma integration.

  python test_gemma.py          offline tests (no API key, no network): fallback, error handling,
                                JSON parsing, and proof that numbers come from Pandas, not the model
  python test_gemma.py --live   additionally makes REAL calls to gemma-4-26b-a4b-it
                                (requires GEMINI_API_KEY) and prints latency + the rewritten question
"""
import os
import sys
import pandas as pd

import llm_router
from llm_router import analyze_with_llm, build_schema_text, _parse_reply

df = pd.DataFrame({
    "Building": ["Block A", "Block B", "Block C", "Block A", "Block B", "Block C"],
    "Electricity (kWh)": [120, 960, 1340, 180, 240, 1200],
    "Date": ["2025-04-01"] * 3 + ["2025-04-02"] * 3,
})
# Block C = 2,540 kWh is the true answer to "which building uses the most".


def fake_model(reply):
    def _call(prompt):
        if isinstance(reply, Exception):
            raise reply
        return reply
    return _call


def run_offline():
    original = llm_router._call_model

    # 1. Schema sent to the model contains names/categories/dates but no numeric cell values.
    schema = build_schema_text(df)
    assert "Electricity (kWh)" in schema and "Block C" in schema and "2025-04-01" in schema
    for v in ("1340", "960", "1200"):
        assert v not in schema, f"numeric value {v} leaked into the prompt"
    print("1 OK: schema has no numeric values")

    # 2. Parsing: fenced JSON, junk, bad status, empty rewrite.
    assert _parse_reply('```json\n{"status":"ok","rewritten_question":"What is the total x?","message":""}\n```')["status"] == "ok"
    assert _parse_reply("sorry, no json") is None
    assert _parse_reply('{"status":"maybe"}') is None
    assert _parse_reply('{"status":"ok","rewritten_question":""}') is None
    assert _parse_reply('{"status":"unanswerable","message":""}') is None
    print("2 OK: reply parsing is strict")

    # 3. Happy path: model rewrites a loose question; the ENGINE computes the number.
    llm_router._call_model = fake_model('{"status":"ok","rewritten_question":"Which Building has the highest Electricity (kWh)?","message":""}')
    r = analyze_with_llm(df, "which block is the biggest power hog?")
    assert r["success"] and "Block C" in r["answer"] and "2,540" in r["answer"], r
    assert r["llm"]["used"] is True and r["llm"]["interpreted_question"].startswith("Which Building")
    assert r["question"] == "which block is the biggest power hog?"
    print("3 OK: Gemma rewrite -> engine answer:", r["answer"])

    # 4. The model cannot inject numbers: a lying 'message' on an ok result is never shown.
    llm_router._call_model = fake_model('{"status":"ok","rewritten_question":"What is the total Electricity (kWh)?","message":"It is 999999"}')
    r = analyze_with_llm(df, "total power?")
    assert "4,040" in r["answer"] and "999999" not in r["answer"], r
    print("4 OK: numbers come from Pandas only:", r["answer"])

    # 5. Unanswerable and clarify are passed through as failures with the model's explanation.
    llm_router._call_model = fake_model('{"status":"unanswerable","rewritten_question":"","message":"There is no water column. Available: Electricity (kWh)."}')
    r = analyze_with_llm(df, "what is the water usage?")
    assert r["success"] is False and r["operation"] == "llm_unanswerable" and "water" in r["answer"]
    llm_router._call_model = fake_model('{"status":"clarify","rewritten_question":"","message":"Which metric do you mean?"}')
    r = analyze_with_llm(df, "average?")
    assert r["success"] is False and r["operation"] == "clarification_needed"
    print("5 OK: unanswerable / clarify handled")

    # 6. Date questions survive the rewrite (engine still applies the period filter).
    llm_router._call_model = fake_model('{"status":"ok","rewritten_question":"What is the total Electricity (kWh) last month?","message":""}')
    r = analyze_with_llm(df, "total power last month")
    assert r["success"] is False and r["operation"] == "date_filter_no_records", r
    import datetime
    first_this = datetime.date.today().replace(day=1)
    last_month_day = (first_this - datetime.timedelta(days=1)).replace(day=10)
    recent = pd.DataFrame({"Building": ["Block A", "Block B"], "Electricity (kWh)": [100, 400],
                           "Date": [last_month_day.isoformat(), first_this.isoformat()]})
    r = analyze_with_llm(recent, "total power last month")
    assert r["success"] and "100" in r["answer"] and "500" not in r["answer"] and r["period"]["records_in_period"] == 1, r
    print("6 OK: relative-date filtering still done by the engine:", r["answer"])

    # 7. API failures -> graceful fallback to the existing engine, with the reason recorded.
    for exc, expect in [
        (RuntimeError("GEMINI_API_KEY is not set"), "GEMINI_API_KEY"),
        (TimeoutError("timed out"), "could not be reached"),
    ]:
        llm_router._call_model = fake_model(exc)
        r = analyze_with_llm(df, "What is the total electricity consumption?")
        assert r["success"] and "4,040" in r["answer"], r
        assert r["llm"]["used"] is False and r["llm"]["fallback"] and expect in r["llm"]["error"]
    llm_router._call_model = fake_model("not json at all")
    r = analyze_with_llm(df, "What is the total electricity consumption?")
    assert r["success"] and r["llm"]["used"] is False
    print("7 OK: API errors / bad replies fall back to the built-in engine")

    # 8. Failed question + failed API -> user is told why.
    llm_router._call_model = fake_model(RuntimeError("GEMINI_API_KEY is not set"))
    r = analyze_with_llm(df, "Tell me a joke about dogs")
    assert r["success"] is False and "AI interpreter was unavailable" in r["answer"]
    print("8 OK: failure message explains the interpreter was unavailable")

    llm_router._call_model = original
    print("\nALL OFFLINE GEMMA TESTS PASSED")


def run_live():
    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not set - skipping live test")
        return
    print(f"\nLIVE test against model: {llm_router.MODEL}")
    questions = [
        "which block is the biggest power hog?",
        "how much electricity did we burn in total?",
        "what is the water usage?",
    ]
    for q in questions:
        r = analyze_with_llm(df, q)
        print(f"\nQ: {q}\n  answer : {r['answer']}\n  gemma  : {r['llm']}")
        assert r["llm"]["used"] is True, f"Gemma was NOT used: {r['llm']['error']}"
        assert r["llm"]["latency_ms"] is not None
    print("\nLIVE TEST PASSED: Gemma was called and the engine produced the numbers")


if __name__ == "__main__":
    run_offline()
    if "--live" in sys.argv:
        run_live()
