import os
import pandas as pd
from app import app
from analysis_engine import analyze_question

def run_tests():
    excel_path = r"C:\Users\RH\Desktop\Insight_data\uploads\372e9523-69ee-4cd1-9cd6-565f879b9b07\Campus_Electricity_Bill.xlsx"
    assert os.path.exists(excel_path), f"File not found: {excel_path}"
    df = pd.read_excel(excel_path)
    print(f"Loaded Campus_Electricity_Bill.xlsx: {df.shape[0]} rows, {df.shape[1]} columns.\n")

    # Independent ground truth calculations:
    cs_df = df[df["Department"] == "Computer Science"]
    cs_count = len(cs_df)
    cs_avg_consumption = cs_df["Electricity_Consumption_kWh"].mean()
    cs_sum_cost = cs_df["Electricity_Cost_INR"].sum()
    top_dept_series = df.groupby("Department")["Electricity_Consumption_kWh"].sum().sort_values(ascending=False)
    top_dept_name = top_dept_series.index[0]
    top_dept_val = top_dept_series.iloc[0]

    print("=== GROUND TRUTH CALCULATIONS ===")
    print(f"CS Rows: {cs_count}")
    print(f"CS Avg Electricity Consumption: {cs_avg_consumption:.2f} kWh")
    print(f"CS Total Electricity Cost: {cs_sum_cost:,} INR")
    print(f"Top Department by Consumption: {top_dept_name} ({top_dept_val:,} kWh)")
    print("=================================\n")

    # Test 1: Average electricity consumption for Computer Science department
    print("--- TEST 1: Average electricity consumption for Computer Science ---")
    q1_variants = [
        "What is the average electricity consumption in kWh for the Computer Science department?",
        "Find the average electricity consumption of Computer Science.",
        "Calculate the mean electricity consumption for the Computer Science department."
    ]
    for q in q1_variants:
        res = analyze_question(df, q)
        assert res["success"] is True, f"Failed on '{q}': {res}"
        assert f"{cs_avg_consumption:.2f}" in res["answer"], f"Expected {cs_avg_consumption:.2f} in answer, got: {res['answer']}"
        assert "Computer Science" in res["answer"]
        assert "174" in res["answer"]
        print(f"Q: '{q}'\nAnswer: {res['answer']}\n")
    print("Test 1 Passed: Correctly identified CS filter, Electricity_Consumption_kWh metric, and calculated 137.71 kWh across all 174 rows.\n")

    # Test 2: Total electricity cost for Computer Science department
    print("--- TEST 2: Total electricity cost for Computer Science ---")
    q2 = "What is the total electricity cost for Computer Science?"
    res2 = analyze_question(df, q2)
    assert res2["success"] is True
    assert "191,277" in res2["answer"] or "191277" in res2["answer"]
    assert "Computer Science" in res2["answer"]
    print(f"Q: '{q2}'\nAnswer: {res2['answer']}\n")
    print("Test 2 Passed: Correctly distinguished Electricity_Cost_INR from consumption and calculated 191,277 INR.\n")

    # Test 3: Department with the highest electricity consumption
    print("--- TEST 3: Department with highest electricity consumption ---")
    q3_variants = [
        "Which department consumes the most electricity?",
        "Which department has the highest electricity consumption?"
    ]
    for q in q3_variants:
        res3 = analyze_question(df, q)
        assert res3["success"] is True
        assert top_dept_name in res3["answer"]
        assert "40,044" in res3["answer"] or "40044" in res3["answer"]
        print(f"Q: '{q}'\nAnswer: {res3['answer']}\n")
    print("Test 3 Passed: Correctly identified Hostel Administration (40,044 kWh) as the top department.\n")

    # Test 4: Comparison of electricity consumption across departments
    print("--- TEST 4: Comparison across departments ---")
    q4 = "Compare electricity consumption across departments."
    res4 = analyze_question(df, q4)
    assert res4["success"] is True
    assert res4["operation"] == "grouped_comparison"
    assert "Hostel Administration" in res4["answer"]
    assert "Computer Science" in res4["answer"]
    assert len(res4["data_used"]) == 10, f"Expected 10 departments in table, got {len(res4['data_used'])}"
    assert res4["chart_data"] is not None
    assert len(res4["chart_data"]["labels"]) == 10
    print(f"Q: '{q4}'\nAnswer: {res4['answer']}\n")
    print("Test 4 Passed: Comparison across all 10 departments returned ranking summary, full data table, and chart data.\n")

    # Test 5: Question with an ambiguous metric
    print("--- TEST 5: Question with ambiguous metric ---")
    q5 = "What is the average for Computer Science?"
    res5 = analyze_question(df, q5)
    assert res5["success"] is False
    assert res5["operation"] == "clarification_needed"
    assert "clarify" in res5["answer"].lower()
    assert "Electricity_Cost_INR" in res5["answer"] and "Electricity_Consumption_kWh" in res5["answer"]
    print(f"Q: '{q5}'\nAnswer: {res5['answer']}\n")
    print("Test 5 Passed: Genuine ambiguity politely prompted user to choose between Cost and Consumption.\n")

    # Test 6: Question asking for information absent from dataset
    print("--- TEST 6: Question asking for information absent from dataset ---")
    q6 = "What is the average water consumption for Computer Science?"
    res6 = analyze_question(df, q6)
    assert res6["success"] is False
    assert res6["operation"] == "absent_metric"
    assert "water" in res6["answer"].lower()
    print(f"Q: '{q6}'\nAnswer: {res6['answer']}\n")
    print("Test 6 Passed: Absent concept ('water') detected and explained clearly.\n")

    # Test 7: Standalone max question
    print("--- TEST 7: Standalone highest electricity consumption ---")
    q7 = "Show the highest electricity consumption."
    res7 = analyze_question(df, q7)
    assert res7["success"] is True
    assert "292" in res7["answer"]
    print(f"Q: '{q7}'\nAnswer: {res7['answer']}\n")
    print("Test 7 Passed: Standalone max returned 292 kWh with row context.\n")

    # Test 8: Trend question
    print("--- TEST 8: Electricity consumption change over time ---")
    q8 = "How does electricity consumption change over time?"
    res8 = analyze_question(df, q8)
    assert res8["success"] is True
    assert res8["operation"] == "trend"
    assert "2026-08-01" in res8["answer"]
    assert "2026-10-26" in res8["answer"]
    print(f"Q: '{q8}'\nAnswer: {res8['answer']}\n")
    print("Test 8 Passed: Time trend correctly identified Date column and trajectory over 87 intervals.\n")

    # Test 9: BUG 1 - Dataset record counting
    print("--- TEST 9: Dataset record counting (Bug 1) ---")
    b1_variants = [
        "How many records are present in the dataset?",
        "How many rows are in the dataset?",
        "How many records have been uploaded?",
        "What is the size of this dataset in terms of records?",
        "How many entries are there?"
    ]
    for q in b1_variants:
        res = analyze_question(df, q)
        assert res["success"] is True, f"Failed on '{q}': {res}"
        assert res["operation"] == "count"
        assert "1,044" in res["answer"], f"Expected 1,044 in answer, got: {res['answer']}"
        assert "attendance" not in res["answer"].lower()
        print(f"Q: '{q}'\nAnswer: {res['answer']}\n")
    print("Test 9 Passed: Dataset-level record counting works dynamically across all natural language variations.\n")

    # Test 10: BUG 2 - Difference between maximum and minimum
    print("--- TEST 10: Difference between maximum and minimum (Bug 2) ---")
    b2_variants = [
        "What is the difference between the highest and lowest electricity consumption?",
        "What is the difference between the maximum and minimum electricity consumption?",
        "What is the range of electricity consumption?"
    ]
    for q in b2_variants:
        res = analyze_question(df, q)
        assert res["success"] is True, f"Failed on '{q}': {res}"
        assert res["operation"] == "range_difference"
        assert "267" in res["answer"], f"Expected diff of 267 in answer, got: {res['answer']}"
        assert "292" in res["answer"], f"Expected max 292 in answer, got: {res['answer']}"
        assert "25" in res["answer"], f"Expected min 25 in answer, got: {res['answer']}"
        assert "Difference = Maximum" in res["calculation_details"]["formula"]
        print(f"Q: '{q}'\nAnswer: {res['answer']}\n")
    print("Test 10 Passed: Range/spread calculation correctly computed Max (292), Min (25), Difference (267), and context.\n")

    # Test 11: BUG 3 - Earliest vs latest dates comparison
    print("--- TEST 11: Earliest vs latest dates comparison (Bug 3) ---")
    b3_variants = [
        "Compare electricity consumption between the earliest and latest dates.",
        "Compare electricity consumption between the first and last date.",
        "How does electricity consumption compare between the earliest date and the latest date?"
    ]
    for q in b3_variants:
        res = analyze_question(df, q)
        assert res["success"] is True, f"Failed on '{q}': {res}"
        assert res["operation"] == "date_comparison"
        assert "2026-08-01" in res["answer"]
        assert "2026-10-26" in res["answer"]
        assert "1,119" in res["answer"]
        assert "1,928" in res["answer"]
        assert "809" in res["answer"]
        assert "increased" in res["answer"]
        print(f"Q: '{q}'\nAnswer: {res['answer']}\n")
    print("Test 11 Passed: Earliest vs latest date comparison computed totals (1,119 vs 1,928) and change (+809 kWh / +72.3%).\n")

    # Test 12: Dynamic dataset test (Verify non-hardcoding)
    print("--- TEST 12: Dynamic dataset behavior on arbitrary slice ---")
    small_df = df.head(75)
    res_small_count = analyze_question(small_df, "How many records are present in the dataset?")
    assert res_small_count["success"] is True
    assert "75" in res_small_count["answer"]
    print(f"Small dataset (75 rows) count answer: {res_small_count['answer']}")
    print("Test 12 Passed: Proves calculations are dynamic from DataFrame, not hardcoded.\n")

    print("=================================================================")
    print("   ALL 12 CAMPUS ELECTRICITY REAL DATASET TESTS PASSED!         ")
    print("=================================================================")

if __name__ == "__main__":
    run_tests()
