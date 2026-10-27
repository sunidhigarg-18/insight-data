import io
import os
import pandas as pd
from app import app

client = app.test_client()

def test_question_analysis():
    print("=================================================================")
    print("       TESTING INSIGHTDATA QUESTION ANALYSIS ENGINE              ")
    print("=================================================================")

    # 1. Create a sample dataset with > 10 rows to verify full dataset analysis (not just preview's 5 rows)
    # Block C has max electricity (1,340) in row 3, but let's add 12 rows with varying dates and values
    csv_data = """Building,Electricity (kWh),Water (L),Date
Block A,120,320,2025-04-01
Block B,960,410,2025-04-01
Block C,1340,290,2025-04-01
Block D,760,500,2025-04-01
Block E,1100,360,2025-04-01
Block A,180,300,2025-04-02
Block B,240,390,2025-04-02
Block C,1200,310,2025-04-02
Block D,440,480,2025-04-02
Block E,500,350,2025-04-02
"""
    # Totals across all 10 rows:
    # Electricity sum = 120 + 960 + 1340 + 760 + 1100 + 180 + 240 + 1200 + 440 + 500 = 6,840 kWh
    # Electricity average = 6,840 / 10 = 684.0 kWh
    # By Building (summed across both days):
    # Block A: 120 + 180 = 300
    # Block B: 960 + 240 = 1,200
    # Block C: 1340 + 1200 = 2,540 (HIGHEST)
    # Block D: 760 + 440 = 1,200
    # Block E: 1100 + 500 = 1,600
    # Water sum = 320 + 410 + 290 + 500 + 360 + 300 + 390 + 310 + 480 + 350 = 3,710 L
    # Water average = 3,710 / 10 = 371.0 L

    # Upload Dataset 1
    upload_res = client.post(
        "/upload",
        data={"file": (io.BytesIO(csv_data.encode("utf-8")), "campus_data.csv")},
        content_type="multipart/form-data"
    )
    assert upload_res.status_code == 200, f"Upload failed: {upload_res.data}"
    print("Upload: Successfully uploaded campus_data.csv with 10 rows.")

    # 1. Test Maximum Ranking Question: "Which building consumed the most electricity?"
    print("\n--- Test 1: Ranking (Maximum) ---")
    res = client.post("/ask", json={"question": "Which building consumed the most electricity?"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True, f"Failed: {data}"
    assert "Block C" in data["answer"], f"Expected Block C in answer, got: {data['answer']}"
    assert "2,540" in data["answer"] or "2540" in data["answer"], f"Expected 2,540 in answer, got: {data['answer']}"
    print("Result:", data["answer"])
    print("Calculation method:", data["calculation_details"]["method"])
    print("Passed Test 1: Maximum ranking returned correct category (Block C) and full aggregated value.")

    # 2. Test Average Question: "What is the average electricity consumption?"
    print("\n--- Test 2: Average Calculation ---")
    res = client.post("/ask", json={"question": "What is the average electricity consumption?"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "684" in data["answer"], f"Expected 684 in answer, got: {data['answer']}"
    print("Result:", data["answer"])
    print("Passed Test 2: Average calculation returned 684 across all 10 rows.")

    # 3. Test Sum / Total Question: "What is the total electricity consumption?"
    print("\n--- Test 3: Sum / Total Calculation ---")
    res = client.post("/ask", json={"question": "What is the total electricity consumption?"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "6,840" in data["answer"] or "6840" in data["answer"], f"Expected 6,840 in answer, got: {data['answer']}"
    print("Result:", data["answer"])
    print("Passed Test 3: Total sum returned 6,840 kWh across all 10 rows.")

    # 4. Test Comparison Question: "Compare electricity between Block A and Block B."
    print("\n--- Test 4: Comparison ---")
    res = client.post("/ask", json={"question": "Compare electricity usage between Block A and Block B"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    # Block B is 1,200, Block A is 300, difference is 900
    assert "Block B" in data["answer"] and "Block A" in data["answer"]
    assert "900" in data["answer"], f"Expected difference of 900 in answer, got: {data['answer']}"
    print("Result:", data["answer"])
    print("Passed Test 4: Comparison correctly computed difference of 900 between Block A and B.")

    # 5. Test Trend Question: "Show me the trend in electricity consumption over time."
    print("\n--- Test 5: Trend Analysis ---")
    res = client.post("/ask", json={"question": "Show me the trend in electricity consumption over time"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["operation"] == "trend"
    assert "2025-04-01" in data["answer"] and "2025-04-02" in data["answer"]
    print("Result:", data["answer"])
    print("Passed Test 5: Trend analysis correctly grouped dates and computed trajectory.")

    # 6. Test Ambiguous Question / Clarification: "What is the average?" (multiple numeric cols, none specified)
    print("\n--- Test 6: Ambiguous Question Handling ---")
    res = client.post("/ask", json={"question": "What is the average?"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is False
    assert "clarify" in data["answer"].lower()
    print("Result:", data["answer"])
    print("Passed Test 6: Ambiguous question asked user for clarification with available columns.")

    # 7. Test Unsupported Question
    print("\n--- Test 7: Unsupported Question Handling ---")
    res = client.post("/ask", json={"question": "Tell me a joke about dogs"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is False
    assert data["operation"] == "unsupported"
    print("Result:", data["answer"])
    print("Passed Test 7: Unsupported question returned polite clarification of capabilities.")

    # 8. Test Dynamic Switching: Upload a completely different dataset (e.g. Sales by Department)
    print("\n--- Test 8: Dynamic Dataset Replacement & Full Dataset Recalculation ---")
    sales_csv = """Department,Revenue,Expenses,Quarter
Engineering,500000,120000,Q1
Marketing,250000,80000,Q1
Sales,800000,150000,Q1
Operations,180000,90000,Q1
Engineering,600000,130000,Q2
Marketing,300000,95000,Q2
Sales,950000,160000,Q2
Operations,200000,95000,Q2
"""
    # Sales Total Revenue = 800k + 950k = 1,750,000 (Highest)
    upload_res2 = client.post(
        "/upload",
        data={"file": (io.BytesIO(sales_csv.encode("utf-8")), "company_finances.csv")},
        content_type="multipart/form-data"
    )
    assert upload_res2.status_code == 200
    print("Upload: Successfully replaced dataset with company_finances.csv.")

    res = client.post("/ask", json={"question": "Which department has the highest revenue?"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "Sales" in data["answer"], f"Expected Sales in answer, got: {data['answer']}"
    assert "1,750,000" in data["answer"] or "1750000" in data["answer"], f"Expected 1,750,000 in answer, got: {data['answer']}"
    print("Result:", data["answer"])
    print("Passed Test 8: New dataset successfully replaced old dataset and calculations immediately adapted to new columns (Revenue, Department).")

    print("\n=================================================================")
    print("       ALL QUESTION ANALYSIS ENGINE TESTS PASSED!               ")
    print("=================================================================")

if __name__ == "__main__":
    test_question_analysis()
