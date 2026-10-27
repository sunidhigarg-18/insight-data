import io
import os
import pandas as pd
from app import app

client = app.test_client()

def run_tests():
    print("=== TEST 1: Main page and Logo ===")
    res = client.get("/")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    html = res.data.decode("utf-8")
    assert "static/images/logo.png" in html, "Logo image path missing from index.html"
    assert "Insight<span class=\"brand-highlight\">Data</span>" in html, "Brand text missing or incorrect"
    assert os.path.exists(os.path.join("static", "images", "logo.png")), "logo.png file does not exist"
    assert "Supported formats: CSV, Excel (.csv, .xlsx, .xls)" in html, "Supported formats text missing from index.html"
    print("Test 1 Passed: Main page, logo, and supported formats hint verified.")

    print("\n=== TEST 2: Valid CSV Upload ===")
    csv_content = """Building,Electricity (kWh),Water (L),Date
Block A,1250,320,2025-04-01
Block B,960,410,2025-04-01
Block C,760,290,2025-04-01
Block D,1340,500,2025-04-01
Block E,1100,360,2025-04-01
Block F,880,310,2025-04-01
"""
    data = {
        "file": (io.BytesIO(csv_content.encode("utf-8")), "campus_energy_water.csv")
    }
    res = client.post("/upload", data=data, content_type="multipart/form-data")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.data}"
    json_data = res.get_json()
    assert json_data["success"] is True
    ds = json_data["dataset"]
    assert ds["filename"] == "campus_energy_water.csv"
    assert ds["file_type"] == "CSV"
    assert ds["rows"] == 6
    assert ds["cols"] == 4
    assert ds["columns"] == ["Building", "Electricity (kWh)", "Water (L)", "Date"]
    assert len(ds["preview"]) == 5, f"Expected 5 preview rows, got {len(ds['preview'])}"
    assert ds["preview"][0]["Building"] == "Block A"
    print("Test 2 Passed: Valid CSV parsed and verified with 5 preview rows.")

    print("\n=== TEST 3: Valid Excel (.xlsx) Upload ===")
    excel_df = pd.DataFrame({
        "Zone": ["North", "South", "East", "West"],
        "Energy_MW": [45.2, 58.7, 39.1, 62.4],
        "Status": ["Active", "Active", "Maintenance", "Active"]
    })
    excel_buf = io.BytesIO()
    excel_df.to_excel(excel_buf, index=False)
    excel_buf.seek(0)
    res = client.post("/upload", data={"file": (excel_buf, "zone_energy.xlsx")}, content_type="multipart/form-data")
    assert res.status_code == 200, f"Excel upload failed: {res.data}"
    json_data = res.get_json()
    assert json_data["success"] is True
    ds = json_data["dataset"]
    assert ds["filename"] == "zone_energy.xlsx"
    assert ds["file_type"] == "XLSX"
    assert ds["rows"] == 4
    assert ds["cols"] == 3
    print("Test 3 Passed: Valid Excel upload verified.")

    print("\n=== TEST 4: Invalid File Extension ===")
    res = client.post("/upload", data={"file": (io.BytesIO(b"malicious script or text"), "notes.txt")}, content_type="multipart/form-data")
    assert res.status_code == 400
    json_data = res.get_json()
    assert json_data["success"] is False
    assert json_data["error"] == "Unsupported file type. Please upload a CSV or Excel file."
    print("Test 4 Passed: Invalid extension rejected with exact message:", json_data["error"])

    print("\n=== TEST 5: Empty File (0 bytes) ===")
    res = client.post("/upload", data={"file": (io.BytesIO(b""), "empty.csv")}, content_type="multipart/form-data")
    assert res.status_code == 400
    json_data = res.get_json()
    assert json_data["success"] is False
    assert "empty" in json_data["error"]
    print("Test 5 Passed: Empty file rejected with clear message:", json_data["error"])

    print("\n=== TEST 6: Previous Dataset Preservation on Upload Failure ===")
    # The active dataset should still be zone_energy.xlsx from Test 3
    res = client.get("/api/active_dataset")
    assert res.status_code == 200
    active_data = res.get_json()
    assert active_data["has_dataset"] is True
    assert active_data["dataset"]["filename"] == "zone_energy.xlsx"
    print("Test 6 Passed: Active dataset remained 'zone_energy.xlsx' after failed uploads.")

    print("\n=======================================================")
    print("   ALL 6 SYSTEM AND UPLOAD INTEGRATION TESTS PASSED!   ")
    print("=======================================================")

if __name__ == "__main__":
    run_tests()
