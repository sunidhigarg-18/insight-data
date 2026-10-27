import os
import io
import uuid
import datetime
from werkzeug.utils import secure_filename
import pandas as pd
from flask import Flask, render_template, request, jsonify, session

app = Flask(__name__)
# Ensure consistent session handling
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "insightdata-secret-key-2026")

# Limit upload file size to 16 MB
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
UPLOAD_BASE_DIR = os.path.join(app.root_path, "uploads")
os.makedirs(UPLOAD_BASE_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}

def get_session_dir():
    """Returns the dedicated upload directory for the current user session."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    session_dir = os.path.join(UPLOAD_BASE_DIR, session["session_id"])
    os.makedirs(session_dir, exist_ok=True)
    return session_dir

def format_file_size(size_in_bytes):
    """Formats file size into human-readable B, KB, or MB."""
    if size_in_bytes < 1024:
        return f"{size_in_bytes} B"
    elif size_in_bytes < 1024 * 1024:
        return f"{size_in_bytes / 1024:.1f} KB"
    else:
        return f"{size_in_bytes / (1024 * 1024):.1f} MB"

def sanitize_dataframe_preview(df, max_rows=5):
    """
    Safely serializes DataFrame preview to JSON-safe structures,
    formatting dates and numbers cleanly while handling nulls.
    """
    preview_df = df.head(max_rows)
    columns = [str(col) for col in df.columns]
    
    rows = []
    for _, row in preview_df.iterrows():
        row_dict = {}
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                row_dict[str(col)] = "—"
            elif isinstance(val, (pd.Timestamp, datetime.date, datetime.datetime)):
                row_dict[str(col)] = val.strftime("%Y-%m-%d")
            elif isinstance(val, (int, float)):
                if isinstance(val, float) and val.is_integer():
                    row_dict[str(col)] = int(val)
                elif isinstance(val, float):
                    row_dict[str(col)] = round(val, 4)
                else:
                    row_dict[str(col)] = val
            else:
                row_dict[str(col)] = str(val)
        rows.append(row_dict)
    
    return columns, rows

@app.route("/")
def index():
    """Renders the main application view."""
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_dataset():
    """
    Handles CSV and Excel file uploads.
    Validates file format, readability, and content before activating it.
    If validation fails, any existing valid dataset is strictly preserved.
    """
    # 1. Check if file is provided in request
    if "file" not in request.files:
        return jsonify({
            "success": False,
            "error": "No file was attached to the upload request."
        }), 400

    file = request.files["file"]
    if not file or not file.filename.strip():
        return jsonify({
            "success": False,
            "error": "No file selected. Please choose a CSV or Excel file."
        }), 400

    filename = secure_filename(file.filename)
    if not filename:
        filename = "dataset.csv"

    # 2. Check extension
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            "success": False,
            "error": "Unsupported file type. Please upload a CSV or Excel file."
        }), 400

    # 3. Read file content safely in memory first (to protect active dataset from corruption)
    try:
        file_bytes = file.read()
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error reading file upload: {str(e)}"
        }), 400

    if len(file_bytes) == 0:
        return jsonify({
            "success": False,
            "error": "The uploaded file is empty (0 bytes). Please upload a valid dataset."
        }), 400

    # 4. Parse using Pandas
    try:
        file_buffer = io.BytesIO(file_bytes)
        if ext == "csv":
            try:
                df = pd.read_csv(file_buffer)
            except UnicodeDecodeError:
                file_buffer.seek(0)
                df = pd.read_csv(file_buffer, encoding="latin-1")
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(file_buffer)
        else:
            return jsonify({
                "success": False,
                "error": "Unsupported file type. Please upload a CSV or Excel file."
            }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to parse dataset: {str(e)}"
        }), 400

    # 5. Check if dataset has rows and columns
    if df is None or df.empty or len(df.columns) == 0:
        return jsonify({
            "success": False,
            "error": "The uploaded dataset contains no rows or columns."
        }), 400

    # 6. Everything is validated! Now safely persist to user's session storage
    session_dir = get_session_dir()

    # Clear previous active file in user's isolated session folder
    for existing_file in os.listdir(session_dir):
        existing_path = os.path.join(session_dir, existing_file)
        try:
            if os.path.isfile(existing_path):
                os.remove(existing_path)
        except OSError:
            pass

    # Save newly validated file to session storage
    saved_path = os.path.join(session_dir, filename)
    with open(saved_path, "wb") as f:
        f.write(file_bytes)

    # Prepare preview and metadata
    columns, preview_rows = sanitize_dataframe_preview(df, max_rows=5)
    file_size_formatted = format_file_size(len(file_bytes))

    dataset_info = {
        "filename": filename,
        "file_type": ext.upper(),
        "file_size": file_size_formatted,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": columns,
        "preview": preview_rows
    }

    # Store metadata in Flask session
    session["active_dataset"] = dataset_info

    return jsonify({
        "success": True,
        "message": "Dataset uploaded successfully",
        "dataset": dataset_info
    })

@app.route("/api/active_dataset", methods=["GET"])
def get_active_dataset():
    """Returns the currently active dataset for this session if one exists."""
    active = session.get("active_dataset")
    if active:
        return jsonify({
            "has_dataset": True,
            "dataset": active
        })
    return jsonify({
        "has_dataset": False
    })

def load_active_dataframe():
    """Loads the active DataFrame from user session storage."""
    active = session.get("active_dataset")
    if not active or "session_id" not in session:
        return None, "No active dataset found. Please upload a CSV or Excel dataset first."

    session_dir = os.path.join(UPLOAD_BASE_DIR, session["session_id"])
    filename = active.get("filename")
    if not filename:
        return None, "Active dataset metadata is missing."

    file_path = os.path.join(session_dir, filename)
    if not os.path.exists(file_path):
        return None, f"Dataset file '{filename}' was not found on the server. Please re-upload it."

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    try:
        if ext == "csv":
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding="latin-1")
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(file_path)
        else:
            return None, f"Unsupported file type: .{ext}"
        return df, None
    except Exception as e:
        return None, f"Error reading active dataset: {str(e)}"

@app.route("/ask", methods=["POST"])
def ask_question():
    """
    Receives user question in plain English, runs analysis against the full active dataset,
    and returns deterministic calculations with transparent steps.
    """
    data = request.get_json(silent=True) or request.form
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({
            "success": False,
            "error": "No question was provided. Please type a question about your data."
        }), 400

    df, err = load_active_dataframe()
    if err:
        return jsonify({
            "success": False,
            "error": err,
            "answer": err
        }), 400

    # Gemma 4 interprets the question; all calculations still run in analysis_engine (Pandas).
    from llm_router import analyze_with_llm
    result = analyze_with_llm(df, question)
    return jsonify(result), 200

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "success": False,
        "error": "The uploaded file exceeds the 16 MB size limit."
    }), 413

if __name__ == "__main__":
    print("Starting InsightData server at http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)
