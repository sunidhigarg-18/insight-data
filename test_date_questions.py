import datetime
import pandas as pd
from analysis_engine import analyze_question, resolve_relative_period, find_relative_period

TODAY = datetime.date(2026, 10, 26)  # Monday

def run_tests():
    # --- period resolution ---
    def span(q):
        s, e, _ = resolve_relative_period(find_relative_period(q), TODAY)
        return s.isoformat(), e.isoformat()
    assert span("last month") == ("2026-09-01", "2026-09-30")
    assert span("this month") == ("2026-10-01", "2026-10-31")
    assert span("this week") == ("2026-10-26", "2026-11-01")
    assert span("last week") == ("2026-10-19", "2026-10-25")
    assert span("last quarter") == ("2026-07-01", "2026-09-30")
    assert span("this quarter") == ("2026-10-01", "2026-12-31")
    assert span("last year") == ("2025-01-01", "2025-12-31")
    assert span("yesterday") == ("2026-10-25", "2026-10-25")
    assert span("last 7 days") == ("2026-10-20", "2026-10-26")
    assert resolve_relative_period(find_relative_period("last quarter"), datetime.date(2026, 1, 15))[:2] == (datetime.date(2025, 10, 1), datetime.date(2025, 12, 31))
    assert resolve_relative_period(find_relative_period("last month"), datetime.date(2026, 1, 15))[:2] == (datetime.date(2025, 12, 1), datetime.date(2025, 12, 31))
    assert find_relative_period("between the first and last date") is None
    assert find_relative_period("earliest and latest dates") is None
    print("period resolution OK")

    # --- Example 1: building / electricity / last month ---
    rows = []
    for d, b, kwh in [
        ("2026-08-15", "Block A", 9999), ("2026-09-02", "Block A", 100), ("2026-09-20", "Block A", 150),
        ("2026-09-10", "Block B", 400), ("2026-09-30", "Block C", 120),
        ("2026-10-05", "Block C", 5000),
    ]:
        rows.append({"Building": b, "Electricity (kWh)": kwh, "Date": d})
    df = pd.DataFrame(rows)
    r = analyze_question(df, "Which building used the most electricity last month?", today=TODAY)
    print(r["answer"]); assert r["success"] and "Block B" in r["answer"] and "400" in r["answer"], r
    assert r["period"]["records_in_period"] == 4 and r["chart_data"] is not None

    # --- Example 2: product / this week ---
    df2 = pd.DataFrame({
        "Product": ["Pen", "Book", "Pen", "Bag", "Book"],
        "Units Sold": [5, 3, 7, 2, 50],
        "Sale Date": pd.to_datetime(["2026-10-26", "2026-10-27", "2026-10-28", "2026-10-20", "2026-10-30 15:30"], format="mixed"),
    })
    r = analyze_question(df2, "Which product sold the most this week?", today=TODAY)
    print(r["answer"]); assert r["success"] and "Book" in r["answer"] and "53" in r["answer"], r

    # --- Example 3: total revenue last quarter (Excel-style datetime dtype) ---
    df3 = pd.DataFrame({
        "Date": pd.to_datetime(["2026-06-30", "2026-07-01", "2026-08-15", "2026-09-30", "2026-10-01"]),
        "Revenue": [1000, 200, 300, 400, 5000],
    })
    r = analyze_question(df3, "What was the total revenue last quarter?", today=TODAY)
    print(r["answer"]); assert r["success"] and "900" in r["answer"], r

    # --- No date column -> explanation, no guessing ---
    r = analyze_question(pd.DataFrame({"Building": ["A", "B"], "kWh": [1, 2]}), "Which building used the most electricity last month?", today=TODAY)
    print(r["answer"]); assert not r["success"] and r["operation"] == "date_filter_no_column"

    # --- No matching records -> explanation with available range ---
    r = analyze_question(df3, "What was the total revenue last year?", today=TODAY)
    print(r["answer"]); assert not r["success"] and r["operation"] == "date_filter_no_records" and "2026-06-30" in r["answer"]

    # --- Default today (real datetime) and unchanged non-date behaviour ---
    r = analyze_question(df3, "What is the total revenue?")
    assert r["success"] and "period" not in r and "6,900" in r["answer"], r
    r = analyze_question(df3, "What was the total revenue last month?")  # real today; should explain, not guess
    print(r["answer"]); assert "period" in r or not r["success"]
    print("ALL DATE TESTS PASSED")

if __name__ == "__main__":
    run_tests()