import re
import datetime
import pandas as pd
import numpy as np


def extract_unit(column_name: str) -> str:
    """Extracts unit from parentheses '(kWh)' or underscore suffix '_kWh'."""
    col_str = str(column_name).strip()
    match = re.search(r"\(([^)]+)\)", col_str)
    if match:
        unit = match.group(1).strip()
        if unit not in ("$", "€", "£"):
            return f" {unit}"
        return ""

    parts = col_str.split("_")
    if len(parts) > 1:
        suffix = parts[-1].upper()
        if suffix in ("KWH", "MWH", "WATT", "INR", "USD", "EUR", "L", "KG", "PERCENT", "PCT"):
            return f" {parts[-1]}"
    return ""


def clean_col_label(column_name: str) -> str:
    """Removes units and underscores for clean natural language phrasing."""
    cleaned = re.sub(r"\([^)]+\)", "", str(column_name)).strip()
    parts = cleaned.split("_")
    if len(parts) > 1 and parts[-1].upper() in ("KWH", "MWH", "WATT", "INR", "USD", "EUR", "L", "KG"):
        cleaned = "_".join(parts[:-1])
    cleaned = cleaned.replace("_", " ").strip()
    return cleaned if cleaned else str(column_name)


def format_number(val) -> str:
    """Formats numeric values with commas and appropriate decimal places."""
    if pd.isna(val) or val is None:
        return "N/A"
    try:
        fval = float(val)
        if fval.is_integer():
            return f"{int(fval):,}"
        elif abs(fval) >= 100:
            return f"{fval:,.2f}"
        else:
            return f"{fval:,.2f}"
    except (ValueError, TypeError):
        return str(val)


def tokenize(text: str) -> list:
    """Splits text into alphanumeric lowercase tokens, ignoring underscores and punctuation."""
    return re.findall(r"[a-z0-9]+", str(text).lower())


def identify_columns(df: pd.DataFrame):
    """Categorizes columns in DataFrame into numeric, categorical, and date/time."""
    numeric_cols = []
    categorical_cols = []
    date_cols = []

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
            continue

        lower_col = str(col).lower()
        if any(d_kw in lower_col for d_kw in ["date", "timestamp", "datetime"]):
            sample = df[col].dropna().head(10)
            if not sample.empty:
                try:
                    pd.to_datetime(sample, format="mixed")
                    date_cols.append(col)
                    continue
                except Exception:
                    pass

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            sample = df[col].dropna().head(20)
            if not sample.empty:
                coerced = pd.to_numeric(sample, errors="coerce")
                if coerced.notna().sum() / len(sample) >= 0.7:
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
            else:
                categorical_cols.append(col)

    return numeric_cols, categorical_cols, date_cols


def score_column_match(query: str, col_name: str) -> int:
    """
    Computes a flexible matching score between user query and a column name.
    Handles underscores, spacing, units, and semantic domain terms.
    """
    q_lower = query.lower()
    q_tokens = set(tokenize(q_lower))
    col_tokens = set(tokenize(str(col_name)))

    score = 0

    # 1. Exact phrase match (e.g. "electricity consumption" in query)
    col_phrase = " ".join(tokenize(str(col_name)))
    if col_phrase and col_phrase in q_lower:
        score += 35

    # Clean label phrase match (e.g. without unit suffix like inr/kwh)
    clean_phrase = " ".join(tokenize(clean_col_label(str(col_name))))
    if clean_phrase and clean_phrase in q_lower and clean_phrase != col_phrase:
        score += 25

    # 2. Token overlap
    overlap = q_tokens.intersection(col_tokens)
    score += len(overlap) * 10

    # 3. Domain semantic synonym boosts
    synonyms = {
        "cost": ["cost", "price", "bill", "inr", "spending", "spend", "expense", "expenses", "expenditure", "rupees", "rs"],
        "consumption": ["consumption", "consume", "consumes", "consumed", "kwh", "units", "usage", "energy", "mwh"],
        "water": ["water", "aqua", "litres", "liters", "gallons", "l"],
        "building": ["building", "block", "facility", "structure", "hall"],
        "department": ["department", "dept", "team", "division", "branch"],
        "revenue": ["revenue", "sales", "turnover", "income", "earnings"]
    }

    for syn_key, syn_words in synonyms.items():
        col_has_syn = any(w in col_tokens for w in syn_words)
        q_has_syn = any(w in q_tokens for w in syn_words)
        if col_has_syn and q_has_syn:
            score += 15

    return score


def find_category_filters(query: str, df: pd.DataFrame, categorical_cols: list) -> dict:
    """
    Finds specific categorical values in the query to apply as row filters.
    e.g. 'for the Computer Science department' -> {'Department': 'Computer Science'}
    """
    filters = {}
    q_lower = query.lower()

    for col in categorical_cols:
        unique_vals = [v for v in df[col].dropna().unique() if len(str(v).strip()) > 1]
        # Match longer strings first so "Academic Block A" precedes "Block A"
        sorted_vals = sorted(unique_vals, key=lambda x: len(str(x)), reverse=True)

        for val in sorted_vals:
            val_clean = str(val).strip()
            val_lower = val_clean.lower()
            pattern = r"\b" + re.escape(val_lower) + r"\b"
            if re.search(pattern, q_lower):
                filters[col] = val_clean
                break  # Pick the best matching value for this column

    return filters


def find_comparison_entities(query: str, df: pd.DataFrame, categorical_cols: list) -> tuple:
    """
    Finds if two entities from the same categorical column are being compared.
    e.g. 'Compare Block A and Block B' or 'Computer Science vs Mathematics'
    """
    q_lower = query.lower()
    for col in categorical_cols:
        unique_vals = [v for v in df[col].dropna().unique() if len(str(v).strip()) > 1]
        sorted_vals = sorted(unique_vals, key=lambda x: len(str(x)), reverse=True)
        matched = []
        for val in sorted_vals:
            val_clean = str(val).strip()
            pattern = r"\b" + re.escape(val_clean.lower()) + r"\b"
            if re.search(pattern, q_lower):
                if val_clean not in matched:
                    matched.append(val_clean)
            if len(matched) >= 2:
                return col, matched[0], matched[1]
    return None, None, None


def check_for_absent_metric(query: str, df: pd.DataFrame) -> str:
    """
    Detects if the user asked about an obvious metric that is not in the dataset (e.g. 'water').
    """
    q_tokens = set(tokenize(query))
    common_metrics = {
        "water": ["water", "litres", "liters", "gallons"],
        "salary": ["salary", "wage", "wages", "payroll"],
        "temperature": ["temperature", "temp", "celsius", "fahrenheit"],
        "attendance": ["attendance", "absenteeism"],
        "profit": ["profit", "margin", "margin_percentage"]
    }

    # All column tokens in dataset
    all_col_tokens = set()
    for c in df.columns:
        all_col_tokens.update(tokenize(str(c)))

    for concept, keywords in common_metrics.items():
        if any(kw in q_tokens for kw in keywords):
            if not any(kw in all_col_tokens for kw in keywords):
                return concept
    return None


def is_dataset_count_query(query: str) -> bool:
    """
    Detects if query is asking for the total number of records/rows/entries in the dataset.
    e.g. 'How many records are present in the dataset?'
         'How many rows are in the dataset?'
         'How many records have been uploaded?'
         'What is the size of this dataset in terms of records?'
         'How many entries are there?'
    """
    q_lower = query.lower()
    q_tokens = set(tokenize(q_lower))

    count_phrases = [
        "how many records", "how many rows", "how many entries", "how many data points",
        "number of records", "number of rows", "number of entries",
        "total records", "total rows", "total entries", "total number of records",
        "count of records", "count of rows", "count of entries", "count records",
        "record count", "row count", "entry count",
        "records are present", "rows are present", "entries are present",
        "records present", "rows present", "entries present",
        "records have been uploaded", "rows have been uploaded", "records uploaded",
        "size of this dataset", "size of the dataset", "dataset size"
    ]
    if any(p in q_lower for p in count_phrases):
        return True

    if ("how many" in q_lower or "count" in q_tokens) and any(w in q_tokens for w in ["records", "rows", "entries"]):
        return True

    if "size" in q_tokens and ("dataset" in q_tokens or "records" in q_tokens or "rows" in q_tokens):
        return True

    return False


def _analyze_question_core(df: pd.DataFrame, question: str) -> dict:
    """
    Main analysis entry point.
    Interprets natural-language questions, applies filters, performs calculations on full data,
    and returns structured results.
    """
    if df is None or df.empty:
        return {
            "success": False,
            "question": question,
            "answer": "No dataset is currently active. Please upload a CSV or Excel dataset first.",
            "operation": "none",
            "error": "Dataset is empty or not loaded."
        }

    q = question.strip()
    q_lower = q.lower()

    numeric_cols, categorical_cols, date_cols = identify_columns(df)

    # 0. Check for dataset-level or filtered record count queries (does not require numeric columns)
    if is_dataset_count_query(q):
        matched_filters = find_category_filters(q, df, categorical_cols)
        if matched_filters:
            filter_col, filter_val = list(matched_filters.items())[0]
            filtered_df = df[df[filter_col].astype(str).str.strip().str.lower() == str(filter_val).strip().lower()]
            filter_count = len(filtered_df)
            total_rows = len(df)
            return {
                "success": True,
                "question": question,
                "answer": f"There are {filter_count:,} records for {filter_val} in the dataset (out of {total_rows:,} total records).",
                "operation": "filtered_count",
                "calculation_details": {
                    "method": f"Count records where {filter_col} = '{filter_val}'",
                    "formula": f"count({filter_col} == '{filter_val}') = {filter_count:,}",
                    "steps": [
                        f"1. Filtered dataset by {filter_col} = '{filter_val}'",
                        f"2. Matched {filter_count:,} out of {total_rows:,} total records"
                    ]
                },
                "data_used": [
                    {"Filter": f"{filter_col} = {filter_val}", "Count": f"{filter_count:,}"},
                    {"Metric": "Total Dataset Rows", "Count": f"{total_rows:,}"}
                ],
                "chart_data": None
            }

        total_rows = len(df)
        return {
            "success": True,
            "question": question,
            "answer": f"The dataset contains {total_rows:,} records.",
            "operation": "count",
            "calculation_details": {
                "method": "Count total records in dataset",
                "formula": f"count() = {total_rows:,}",
                "steps": [
                    "1. Evaluated active uploaded dataset",
                    f"2. Total records counted: {total_rows:,}"
                ]
            },
            "data_used": [{"Metric": "Total Records", "Count": f"{total_rows:,}"}],
            "chart_data": None
        }

    if not numeric_cols:
        return {
            "success": False,
            "question": question,
            "answer": "The active dataset does not contain any numeric columns for calculations.",
            "operation": "none",
            "error": "No numeric columns found."
        }

    # 1. Check for general analytical intent
    analysis_keywords = [
        "sum", "total", "average", "mean", "avg", "highest", "lowest", "max", "maximum",
        "min", "minimum", "most", "least", "compare", "difference", "trend", "count",
        "how many", "top", "bottom", "largest", "smallest", "greatest", "versus", " vs ",
        "over time", "change over time", "changes over time", "consumed", "consumption",
        "usage", "cost", "spending", "revenue", "bill", "records", "entries", "size",
        "earliest", "latest", "range", "spread", "gap"
    ]
    has_analysis_intent = any(kw in q_lower for kw in analysis_keywords) or any(clean_col_label(str(c)).lower() in q_lower for c in df.columns)
    if not has_analysis_intent:
        return {
            "success": False,
            "question": question,
            "answer": (
                "I couldn't identify a supported analytical operation in your question. "
                "I can calculate statistics (sum, average, min, max, count), rankings, "
                "comparisons, and time trends based on your dataset."
            ),
            "operation": "unsupported",
            "error": "Question does not ask for a supported analytical calculation."
        }

    # 2. Check if user is asking for a metric that is completely absent from this dataset
    absent_concept = check_for_absent_metric(q, df)
    if absent_concept:
        avail_cols_str = ", ".join(str(c) for c in numeric_cols)
        return {
            "success": False,
            "question": question,
            "answer": f"The dataset does not contain information about {absent_concept}. Available numeric columns: {avail_cols_str}",
            "operation": "absent_metric",
            "error": f"Requested metric '{absent_concept}' is absent from dataset."
        }

    # 3. Detect category filters (e.g. "for Computer Science", "in Academic Block A")
    matched_filters = find_category_filters(q, df, categorical_cols)

    # 4. Score and match target numeric column
    col_scores = {c: score_column_match(q, c) for c in numeric_cols}
    sorted_cols = sorted(col_scores.items(), key=lambda x: x[1], reverse=True)
    best_col, best_score = sorted_cols[0]

    target_num_col = None

    # Count questions don't strictly require a numeric column
    is_count = any(phrase in q_lower for phrase in ["how many records", "how many rows", "number of rows", "total records", "count of records", "record count"])

    if is_count:
        target_num_col = None
    else:
        # Check for ambiguity among numeric columns
        if len(numeric_cols) == 1:
            target_num_col = numeric_cols[0]
        else:
            if best_score == 0:
                # Genuine ambiguity: No specific metric was mentioned
                formatted_cols = ", ".join(str(c) for c in numeric_cols)
                filter_hint = f" for {list(matched_filters.values())[0]}" if matched_filters else ""
                return {
                    "success": False,
                    "question": question,
                    "answer": f"Please clarify which metric you want to analyze{filter_hint}. Available numeric columns: {formatted_cols}",
                    "operation": "clarification_needed",
                    "error": f"Ambiguous metric. Available numeric columns: {formatted_cols}"
                }
            else:
                # If runner up is tied with top score and > 0, it's ambiguous
                if len(sorted_cols) > 1 and sorted_cols[1][1] == best_score and best_score > 0:
                    tied_cols = [c for c, s in sorted_cols if s == best_score]
                    formatted_cols = " or ".join(str(c) for c in tied_cols)
                    return {
                        "success": False,
                        "question": question,
                        "answer": f"Please clarify which metric you want to analyze: {formatted_cols}?",
                        "operation": "clarification_needed",
                        "error": f"Tied metric scores: {formatted_cols}"
                    }
                target_num_col = best_col

    unit_str = extract_unit(target_num_col) if target_num_col else ""
    metric_label = clean_col_label(target_num_col) if target_num_col else "Records"

    # =========================================================================
    # CASE A: QUERY HAS A SPECIFIC CATEGORY FILTER (e.g. for Computer Science)
    # =========================================================================
    if matched_filters and target_num_col:
        filter_col, filter_val = list(matched_filters.items())[0]

        # Apply filter across the entire dataset
        filtered_df = df[df[filter_col].astype(str).str.strip().str.lower() == str(filter_val).strip().lower()]
        filter_count = len(filtered_df)

        if filter_count == 0:
            return {
                "success": False,
                "question": question,
                "answer": f"No records found matching {filter_col} = '{filter_val}'.",
                "operation": "filter_no_results",
                "error": f"Filter returned 0 rows."
            }

        clean_series = pd.to_numeric(filtered_df[target_num_col], errors="coerce").dropna()
        if clean_series.empty:
            return {
                "success": False,
                "question": question,
                "answer": f"No valid numeric data found in '{target_num_col}' for {filter_val}.",
                "operation": "filter_no_data",
                "error": "No valid numeric entries."
            }

        # Check operation for filtered data
        is_avg = any(w in q_lower for w in ["average", "mean", "avg"])
        is_sum = any(w in q_lower for w in ["sum", "total", "overall", "combined", "altogether"])
        is_max = any(w in q_lower for w in ["max", "maximum", "highest", "most", "peak"])
        is_min = any(w in q_lower for w in ["min", "minimum", "lowest", "least"])

        # Default to average if requested, else sum if requested, else max
        if is_avg:
            calc_val = clean_series.mean()
            sum_val = clean_series.sum()
            answer_text = (
                f"The average {metric_label.lower()} for the {filter_val} {filter_col.lower()} "
                f"is {format_number(calc_val)}{unit_str} (calculated across all {filter_count:,} records)."
            )
            calc_details = {
                "method": f"Calculate average of {target_num_col} for {filter_col} = '{filter_val}'",
                "formula": f"Average = {format_number(sum_val)} / {filter_count:,} = {format_number(calc_val)}{unit_str}",
                "steps": [
                    f"1. Filtered dataset where {filter_col} = '{filter_val}' ({filter_count:,} records found)",
                    f"2. Sum of {target_num_col}: {format_number(sum_val)}{unit_str}",
                    f"3. Mean = {format_number(sum_val)} / {filter_count:,} = {format_number(calc_val)}{unit_str}"
                ]
            }
            op_name = "filtered_average"
        elif is_sum:
            calc_val = clean_series.sum()
            answer_text = (
                f"The total {metric_label.lower()} for the {filter_val} {filter_col.lower()} "
                f"is {format_number(calc_val)}{unit_str} across all {filter_count:,} records."
            )
            calc_details = {
                "method": f"Calculate total sum of {target_num_col} for {filter_col} = '{filter_val}'",
                "formula": f"Sum = {format_number(calc_val)}{unit_str}",
                "steps": [
                    f"1. Filtered dataset where {filter_col} = '{filter_val}' ({filter_count:,} records found)",
                    f"2. Summed all entries in '{target_num_col}': {format_number(calc_val)}{unit_str}"
                ]
            }
            op_name = "filtered_sum"
        elif is_max or is_min:
            calc_val = clean_series.max() if is_max else clean_series.min()
            adj = "highest" if is_max else "lowest"
            answer_text = (
                f"The {adj} {metric_label.lower()} for {filter_val} "
                f"is {format_number(calc_val)}{unit_str} across {filter_count:,} records."
            )
            calc_details = {
                "method": f"Find {adj} value in {target_num_col} for {filter_val}",
                "formula": f"{'max' if is_max else 'min'}() = {format_number(calc_val)}{unit_str}",
                "steps": [
                    f"1. Filtered {filter_count:,} records for {filter_val}",
                    f"2. Evaluated {adj} {target_num_col}: {format_number(calc_val)}{unit_str}"
                ]
            }
            op_name = f"filtered_{adj}"
        else:
            # Fallback to mean
            calc_val = clean_series.mean()
            answer_text = f"The average {metric_label.lower()} for {filter_val} is {format_number(calc_val)}{unit_str}."
            calc_details = {
                "method": f"Average of {target_num_col} for {filter_val}",
                "formula": f"Mean = {format_number(calc_val)}{unit_str}",
                "steps": [f"Evaluated {filter_count:,} records for {filter_val}"]
            }
            op_name = "filtered_metric"

        # Prepare supporting sample data
        data_used = [
            {"Filter": f"{filter_col} = {filter_val}", "Records Analyzed": filter_count},
            {"Metric": f"Sum of {metric_label}", "Value": f"{format_number(clean_series.sum())}{unit_str}"},
            {"Metric": f"Average of {metric_label}", "Value": f"{format_number(clean_series.mean())}{unit_str}"},
            {"Metric": f"Min {metric_label}", "Value": f"{format_number(clean_series.min())}{unit_str}"},
            {"Metric": f"Max {metric_label}", "Value": f"{format_number(clean_series.max())}{unit_str}"}
        ]

        return {
            "success": True,
            "question": question,
            "answer": answer_text,
            "operation": op_name,
            "calculation_details": calc_details,
            "data_used": data_used,
            "chart_data": None
        }

    # =========================================================================
    # CASE B: OPERATION 1 - COUNT OF RECORDS
    # =========================================================================
    if is_count:
        total_rows = len(df)
        return {
            "success": True,
            "question": question,
            "answer": f"The dataset contains {total_rows:,} total records.",
            "operation": "count",
            "calculation_details": {
                "method": "Count total records in dataset",
                "formula": f"count() = {total_rows:,}",
                "steps": [f"Counted all rows across dataset: {total_rows:,} records"]
            },
            "data_used": [{"Metric": "Total Rows", "Count": total_rows}],
            "chart_data": None
        }

    # =========================================================================
    # CASE C: OPERATION 2 - TREND OVER TIME
    # =========================================================================
    if any(w in q_lower for w in ["trend", "over time", "timeline", "by date", "monthly", "yearly", "daily", "changes over time", "change over time"]):
        if not date_cols:
            return {
                "success": False,
                "question": question,
                "answer": f"The dataset does not contain a recognized date or time column to calculate a trend over time. Available columns: {', '.join(df.columns)}",
                "operation": "trend",
                "error": "No date or time column found for trend analysis."
            }

        date_col = date_cols[0]
        temp_df = df.copy()
        temp_df["_parsed_date"] = pd.to_datetime(temp_df[date_col], errors="coerce")
        temp_df = temp_df.dropna(subset=["_parsed_date", target_num_col]).sort_values("_parsed_date")

        grouped_trend = temp_df.groupby(temp_df["_parsed_date"].dt.strftime("%Y-%m-%d"))[target_num_col].sum()
        if len(grouped_trend) < 2:
            return {
                "success": False,
                "question": question,
                "answer": f"Not enough distinct time periods in '{date_col}' to calculate a trend.",
                "operation": "trend",
                "error": "Fewer than 2 distinct time periods found."
            }

        start_date = grouped_trend.index[0]
        end_date = grouped_trend.index[-1]
        start_val = grouped_trend.iloc[0]
        end_val = grouped_trend.iloc[-1]
        diff_val = end_val - start_val
        direction = "increased" if diff_val > 0 else "decreased" if diff_val < 0 else "remained constant"
        pct_change = (diff_val / abs(start_val) * 100) if start_val != 0 else 0
        pct_str = f" ({pct_change:+.1f}%)" if start_val != 0 else ""

        answer_text = (
            f"Over the recorded period from {start_date} to {end_date}, {metric_label.lower()} {direction} "
            f"from {format_number(start_val)}{unit_str} to {format_number(end_val)}{unit_str} "
            f"(net change: {format_number(diff_val)}{unit_str}{pct_str})."
        )

        return {
            "success": True,
            "question": question,
            "answer": answer_text,
            "operation": "trend",
            "calculation_details": {
                "method": f"Time-series trend of {target_num_col} along {date_col}",
                "formula": f"Net Change: {format_number(end_val)} - {format_number(start_val)} = {format_number(diff_val)}{unit_str}",
                "steps": [
                    f"1. Sorted {len(temp_df):,} records chronologically along '{date_col}'",
                    f"2. Aggregated into {len(grouped_trend)} distinct time intervals",
                    f"3. Start value ({start_date}): {format_number(start_val)}{unit_str}",
                    f"4. End value ({end_date}): {format_number(end_val)}{unit_str}",
                    f"5. Overall trajectory: {direction.capitalize()} by {format_number(abs(diff_val))}{unit_str}"
                ]
            },
            "data_used": [{"Date": str(d), target_num_col: format_number(v)} for d, v in grouped_trend.items()],
            "chart_data": {
                "type": "line",
                "title": f"{target_num_col} Trend Over Time",
                "labels": list(grouped_trend.index),
                "values": [round(float(v), 2) for v in grouped_trend.values],
                "x_label": str(date_col),
                "y_label": str(target_num_col)
            }
        }

    # =========================================================================
    # CASE C2: OPERATION 2B - EARLIEST VS LATEST DATE COMPARISON
    # =========================================================================
    is_date_comparison = (
        (("earliest" in q_lower or "first date" in q_lower or "start date" in q_lower or "first recorded date" in q_lower) and
         ("latest" in q_lower or "last date" in q_lower or "end date" in q_lower or "last recorded date" in q_lower))
        or ("earliest and latest" in q_lower)
        or ("first and last" in q_lower and ("date" in q_lower or "dates" in q_lower))
    )

    if is_date_comparison:
        if not date_cols:
            return {
                "success": False,
                "question": question,
                "answer": f"The dataset does not contain a recognized date or time column to compare dates. Available columns: {', '.join(df.columns)}",
                "operation": "date_comparison",
                "error": "No date column found."
            }

        date_col = date_cols[0]
        temp_df = df.copy()
        temp_df["_parsed_date"] = pd.to_datetime(temp_df[date_col], errors="coerce")
        temp_df = temp_df.dropna(subset=["_parsed_date", target_num_col])

        if temp_df.empty:
            return {
                "success": False,
                "question": question,
                "answer": f"No valid date entries found in '{date_col}' to compare dates.",
                "operation": "date_comparison",
                "error": "No valid parsed dates."
            }

        temp_df["_date_str"] = temp_df["_parsed_date"].dt.strftime("%Y-%m-%d")
        grouped_by_date = temp_df.groupby("_date_str")[target_num_col].sum().sort_index()

        if len(grouped_by_date) < 2:
            return {
                "success": False,
                "question": question,
                "answer": f"The dataset contains only one distinct date ({grouped_by_date.index[0]}), so a comparison between earliest and latest dates is not possible.",
                "operation": "date_comparison",
                "error": "Fewer than 2 distinct dates found."
            }

        earliest_date = grouped_by_date.index[0]
        latest_date = grouped_by_date.index[-1]
        earliest_val = grouped_by_date.iloc[0]
        latest_val = grouped_by_date.iloc[-1]

        diff_val = latest_val - earliest_val
        abs_diff = abs(diff_val)
        if diff_val > 0:
            direction = "increased"
            dir_prep = "by"
        elif diff_val < 0:
            direction = "decreased"
            dir_prep = "by"
        else:
            direction = "remained unchanged"
            dir_prep = "at"

        pct_change = (diff_val / abs(earliest_val) * 100) if earliest_val != 0 else 0
        pct_str = f" ({pct_change:+.1f}%)" if earliest_val != 0 and diff_val != 0 else ""

        answer_text = (
            f"Comparing {metric_label.lower()} between the earliest date ({earliest_date}) and latest date ({latest_date}): "
            f"total {metric_label.lower()} {direction} {dir_prep} {format_number(abs_diff)}{unit_str}{pct_str}, "
            f"from {format_number(earliest_val)}{unit_str} on {earliest_date} to {format_number(latest_val)}{unit_str} on {latest_date}."
        )

        calc_details = {
            "method": f"Compare '{target_num_col}' between earliest ({earliest_date}) and latest ({latest_date}) dates",
            "formula": f"Latest ({format_number(latest_val)}{unit_str}) - Earliest ({format_number(earliest_val)}{unit_str}) = {diff_val:+,.2f}{unit_str}{pct_str}",
            "steps": [
                f"1. Identified date column '{date_col}' across {len(temp_df):,} records",
                f"2. Earliest date ({earliest_date}): total {format_number(earliest_val)}{unit_str}",
                f"3. Latest date ({latest_date}): total {format_number(latest_val)}{unit_str}",
                f"4. Difference: {format_number(latest_val)} - {format_number(earliest_val)} = {format_number(diff_val)}{unit_str}{pct_str}"
            ]
        }

        data_used = [
            {"Date": str(earliest_date), "Period": "Earliest Date", f"Total {target_num_col}": f"{format_number(earliest_val)}{unit_str}"},
            {"Date": str(latest_date), "Period": "Latest Date", f"Total {target_num_col}": f"{format_number(latest_val)}{unit_str}"},
            {"Date": "Net Difference", "Period": direction.capitalize(), f"Total {target_num_col}": f"{format_number(diff_val)}{unit_str}{pct_str}"}
        ]

        chart_data = {
            "type": "bar",
            "title": f"{target_num_col}: Earliest vs Latest Date",
            "labels": [f"Earliest ({earliest_date})", f"Latest ({latest_date})"],
            "values": [round(float(earliest_val), 2), round(float(latest_val), 2)],
            "highlight_index": 1 if latest_val >= earliest_val else 0,
            "x_label": "Date",
            "y_label": str(target_num_col)
        }

        return {
            "success": True,
            "question": question,
            "answer": answer_text,
            "operation": "date_comparison",
            "calculation_details": calc_details,
            "data_used": data_used,
            "chart_data": chart_data
        }

    # =========================================================================
    # CASE D: OPERATION 3 - COMPARISON ACROSS DEPARTMENTS / GROUPS
    # =========================================================================
    # Matches: "Compare electricity consumption across departments" or "compare by department"
    is_group_comparison = any(phrase in q_lower for phrase in [
        "across departments", "across all departments", "across buildings", "across categories",
        "across all", "compare by", "breakdown by", "compare electricity consumption across"
    ]) or ("compare" in q_lower and any(str(c).lower() in q_lower for c in categorical_cols))

    target_cat_col = None
    for c in categorical_cols:
        c_tokens = tokenize(str(c))
        if any(tok in q_lower for tok in c_tokens):
            target_cat_col = c
            break
    if not target_cat_col and categorical_cols:
        target_cat_col = categorical_cols[0]

    # Explicit 2-entity comparison (e.g. Block A vs Block B)
    cmp_col, ent1, ent2 = find_comparison_entities(q, df, categorical_cols)
    if cmp_col and ent1 and ent2:
        sub1 = df[df[cmp_col] == ent1][target_num_col].dropna()
        sub2 = df[df[cmp_col] == ent2][target_num_col].dropna()

        val1 = sub1.sum()
        val2 = sub2.sum()
        diff = abs(val1 - val2)
        higher_ent = ent1 if val1 >= val2 else ent2
        lower_ent = ent2 if val1 >= val2 else ent1
        pct_diff = (diff / val2 * 100) if val2 != 0 else 0

        answer_text = (
            f"Comparing {ent1} ({format_number(val1)}{unit_str}) and {ent2} ({format_number(val2)}{unit_str}): "
            f"{higher_ent} is higher by {format_number(diff)}{unit_str} ({pct_diff:.1f}% higher than {lower_ent})."
        )

        return {
            "success": True,
            "question": question,
            "answer": answer_text,
            "operation": "comparison",
            "calculation_details": {
                "method": f"Compare '{target_num_col}' between '{ent1}' and '{ent2}'",
                "formula": f"|{format_number(val1)} - {format_number(val2)}| = {format_number(diff)}{unit_str}",
                "steps": [
                    f"1. Sum for {ent1}: {format_number(val1)}{unit_str}",
                    f"2. Sum for {ent2}: {format_number(val2)}{unit_str}",
                    f"3. Difference: {format_number(diff)}{unit_str} ({pct_diff:.1f}%)"
                ]
            },
            "data_used": [
                {cmp_col: str(ent1), target_num_col: format_number(val1)},
                {cmp_col: str(ent2), target_num_col: format_number(val2)}
            ],
            "chart_data": {
                "type": "bar",
                "title": f"{target_num_col} Comparison: {ent1} vs {ent2}",
                "labels": [str(ent1), str(ent2)],
                "values": [round(float(val1), 2), round(float(val2), 2)],
                "highlight_index": 0 if val1 >= val2 else 1,
                "x_label": str(cmp_col),
                "y_label": str(target_num_col)
            }
        }

    if is_group_comparison and target_cat_col:
        grouped = df.groupby(target_cat_col)[target_num_col].sum().sort_values(ascending=False).dropna()
        if not grouped.empty:
            top_cat = grouped.index[0]
            top_val = grouped.iloc[0]
            second_cat = grouped.index[1] if len(grouped) > 1 else None
            second_val = grouped.iloc[1] if len(grouped) > 1 else None
            lowest_cat = grouped.index[-1]
            lowest_val = grouped.iloc[-1]

            second_str = f", followed by {second_cat} at {format_number(second_val)}{unit_str}" if second_cat else ""
            answer_text = (
                f"Comparison of {metric_label.lower()} across all {len(grouped)} {target_cat_col.lower()}s: "
                f"{top_cat} has the highest at {format_number(top_val)}{unit_str}{second_str}, "
                f"while {lowest_cat} has the lowest at {format_number(lowest_val)}{unit_str}."
            )

            data_used = [{target_cat_col: str(k), f"Total {target_num_col}": format_number(v)} for k, v in grouped.items()]
            labels = [str(k) for k in grouped.index]
            values = [round(float(v), 2) for v in grouped.values]

            return {
                "success": True,
                "question": question,
                "answer": answer_text,
                "operation": "grouped_comparison",
                "calculation_details": {
                    "method": f"Grouped comparison of {target_num_col} by {target_cat_col}",
                    "formula": f"Top: {top_cat} ({format_number(top_val)}{unit_str}) vs Lowest: {lowest_cat} ({format_number(lowest_val)}{unit_str})",
                    "steps": [
                        f"1. Aggregated {len(df):,} records across {len(grouped)} {target_cat_col.lower()}s",
                        f"2. Highest: {top_cat} ({format_number(top_val)}{unit_str})",
                        f"3. Lowest: {lowest_cat} ({format_number(lowest_val)}{unit_str})",
                        f"4. Total overall sum: {format_number(grouped.sum())}{unit_str}"
                    ]
                },
                "data_used": data_used,
                "chart_data": {
                    "type": "bar",
                    "title": f"{target_num_col} by {target_cat_col}",
                    "labels": labels,
                    "values": values,
                    "highlight_index": 0,
                    "x_label": str(target_cat_col),
                    "y_label": str(target_num_col)
                }
            }

    # Define operation intent using token matching to prevent substring traps (e.g. 'sum' in 'consumption')
    q_tokens = set(tokenize(q_lower))
    is_avg = bool(q_tokens.intersection({"average", "mean", "avg"}))
    is_sum = bool(q_tokens.intersection({"sum", "total", "altogether", "combined", "overall"}))
    is_highest = bool(q_tokens.intersection({"most", "highest", "max", "maximum", "top", "largest", "greatest", "peak"}))
    is_lowest = bool(q_tokens.intersection({"least", "lowest", "min", "minimum", "bottom", "smallest"}))
    is_diff_extremes = (
        (bool(q_tokens.intersection({"difference", "diff", "gap"})) and 
         bool(q_tokens.intersection({"highest", "max", "maximum", "peak", "top", "greatest", "largest"})) and 
         bool(q_tokens.intersection({"lowest", "min", "minimum", "bottom", "least", "smallest"})))
        or bool(q_tokens.intersection({"range", "spread"}))
    )
    is_ranking_cat = any(w in q_lower for w in ["which", "who", "what department", "what building", "what category"]) or \
                     (target_cat_col and any(tok in q_lower for tok in [f"by {target_cat_col.lower()}", f"across {target_cat_col.lower()}s", f"{target_cat_col.lower()} with the highest", f"{target_cat_col.lower()} with highest", f"{target_cat_col.lower()} with the lowest", f"{target_cat_col.lower()} with lowest"]))

    # =========================================================================
    # CASE D2: OPERATION 3B - DIFFERENCE BETWEEN MAXIMUM AND MINIMUM (RANGE/SPREAD)
    # =========================================================================
    if is_diff_extremes and target_num_col:
        # Check if an explicit categorical column is named (e.g. "between the highest and lowest department")
        explicit_cat_for_range = None
        for c in categorical_cols:
            clean_c = clean_col_label(str(c)).lower()
            if clean_c in q_lower or str(c).lower() in q_lower:
                explicit_cat_for_range = c
                break

        if explicit_cat_for_range:
            grouped = df.groupby(explicit_cat_for_range)[target_num_col].sum().sort_values(ascending=False).dropna()
            if not grouped.empty:
                top_cat = grouped.index[0]
                top_val = grouped.iloc[0]
                lowest_cat = grouped.index[-1]
                lowest_val = grouped.iloc[-1]
                diff_val = top_val - lowest_val

                answer_text = (
                    f"The difference between the highest {explicit_cat_for_range.lower()} ({top_cat} with {format_number(top_val)}{unit_str}) "
                    f"and the lowest {explicit_cat_for_range.lower()} ({lowest_cat} with {format_number(lowest_val)}{unit_str}) "
                    f"is {format_number(diff_val)}{unit_str}."
                )
                calc_details = {
                    "method": f"Difference between highest and lowest {explicit_cat_for_range} for {target_num_col}",
                    "formula": f"Difference = {format_number(top_val)} - {format_number(lowest_val)} = {format_number(diff_val)}{unit_str}",
                    "steps": [
                        f"1. Aggregated {len(df):,} records by {explicit_cat_for_range}",
                        f"2. Highest: {top_cat} ({format_number(top_val)}{unit_str})",
                        f"3. Lowest: {lowest_cat} ({format_number(lowest_val)}{unit_str})",
                        f"4. Difference = {format_number(top_val)} - {format_number(lowest_val)} = {format_number(diff_val)}{unit_str}"
                    ]
                }
                data_used = [
                    {explicit_cat_for_range: str(top_cat), "Extreme": "Highest", target_num_col: f"{format_number(top_val)}{unit_str}"},
                    {explicit_cat_for_range: str(lowest_cat), "Extreme": "Lowest", target_num_col: f"{format_number(lowest_val)}{unit_str}"},
                    {explicit_cat_for_range: "Difference", "Extreme": "Spread", target_num_col: f"{format_number(diff_val)}{unit_str}"}
                ]
                chart_data = {
                    "type": "bar",
                    "title": f"{target_num_col} Spread: {top_cat} vs {lowest_cat}",
                    "labels": [str(top_cat), str(lowest_cat)],
                    "values": [round(float(top_val), 2), round(float(lowest_val), 2)],
                    "highlight_index": 0,
                    "x_label": str(explicit_cat_for_range),
                    "y_label": str(target_num_col)
                }
                return {
                    "success": True,
                    "question": question,
                    "answer": answer_text,
                    "operation": "category_range",
                    "calculation_details": calc_details,
                    "data_used": data_used,
                    "chart_data": chart_data
                }

        # Otherwise, calculate difference across individual dataset records
        clean_series = pd.to_numeric(df[target_num_col], errors="coerce").dropna()
        max_val = clean_series.max()
        min_val = clean_series.min()
        diff_val = max_val - min_val

        max_idx = clean_series.idxmax()
        min_idx = clean_series.idxmin()
        max_row = df.loc[max_idx]
        min_row = df.loc[min_idx]

        max_context = []
        min_context = []
        for c in categorical_cols:
            if c in max_row and pd.notna(max_row[c]):
                max_context.append(str(max_row[c]))
            if c in min_row and pd.notna(min_row[c]):
                min_context.append(str(min_row[c]))

        max_context_str = f" (in {', '.join(max_context[:2])})" if max_context else ""
        min_context_str = f" (in {', '.join(min_context[:2])})" if min_context else ""

        answer_text = (
            f"The difference between the highest and lowest {metric_label.lower()} is {format_number(diff_val)}{unit_str}. "
            f"The highest recorded value is {format_number(max_val)}{unit_str}{max_context_str}, "
            f"and the lowest is {format_number(min_val)}{unit_str}{min_context_str}."
        )

        calc_details = {
            "method": f"Calculate difference between maximum and minimum {target_num_col}",
            "formula": f"Difference = Maximum ({format_number(max_val)}{unit_str}) - Minimum ({format_number(min_val)}{unit_str}) = {format_number(diff_val)}{unit_str}",
            "steps": [
                f"1. Evaluated all {len(clean_series):,} records in '{target_num_col}'",
                f"2. Highest value: {format_number(max_val)}{unit_str}{max_context_str}",
                f"3. Lowest value: {format_number(min_val)}{unit_str}{min_context_str}",
                f"4. Difference = {format_number(max_val)} - {format_number(min_val)} = {format_number(diff_val)}{unit_str}"
            ]
        }

        data_used = [
            {"Record": f"Highest {metric_label}", "Value": f"{format_number(max_val)}{unit_str}", "Details": ", ".join(max_context[:2]) if max_context else "Max record"},
            {"Record": f"Lowest {metric_label}", "Value": f"{format_number(min_val)}{unit_str}", "Details": ", ".join(min_context[:2]) if min_context else "Min record"},
            {"Record": "Difference (Range)", "Value": f"{format_number(diff_val)}{unit_str}", "Details": "Max - Min"}
        ]

        chart_data = {
            "type": "bar",
            "title": f"{metric_label}: Highest vs Lowest",
            "labels": ["Highest", "Lowest"],
            "values": [round(float(max_val), 2), round(float(min_val), 2)],
            "highlight_index": 0,
            "x_label": "Record",
            "y_label": str(target_num_col)
        }

        return {
            "success": True,
            "question": question,
            "answer": answer_text,
            "operation": "range_difference",
            "calculation_details": calc_details,
            "data_used": data_used,
            "chart_data": chart_data
        }

    # =========================================================================
    # CASE E: OPERATION 4 - RANKING (WHICH CATEGORY HAS HIGHEST / LOWEST)
    # =========================================================================
    if (is_highest or is_lowest) and target_cat_col and is_ranking_cat:
        grouped = df.groupby(target_cat_col)[target_num_col].sum().dropna()
        if not grouped.empty:
            best_cat = grouped.idxmax() if is_highest else grouped.idxmin()
            best_val = grouped.loc[best_cat]
            adj = "highest" if is_highest else "lowest"

            if any(w in metric_label.lower() for w in ["consumption", "usage"]):
                m_clean = re.sub(r"\b(consumption|usage)\b", "", metric_label, flags=re.IGNORECASE).strip()
                verb = f"consumed the most {m_clean.lower()}" if is_highest else f"consumed the least {m_clean.lower()}"
            else:
                verb = f"has the highest {metric_label.lower()}" if is_highest else f"has the lowest {metric_label.lower()}"

            answer_text = f"{best_cat} {verb} with {format_number(best_val)}{unit_str}."

            data_used_rows = [
                {target_cat_col: str(k), target_num_col: format_number(v)}
                for k, v in grouped.items()
            ]

            labels = [str(k) for k in grouped.index]
            values = [round(float(v), 2) for v in grouped.values]
            highlight_idx = labels.index(str(best_cat)) if str(best_cat) in labels else 0

            return {
                "success": True,
                "question": question,
                "answer": answer_text,
                "operation": f"ranking_{adj}",
                "calculation_details": {
                    "method": f"Find the {adj} value in {target_num_col} grouped by {target_cat_col}",
                    "formula": f"{'max' if is_highest else 'min'}() -> {format_number(best_val)}{unit_str}",
                    "steps": [
                        f"1. Aggregated {len(df):,} records across {len(grouped)} {target_cat_col} categories",
                        f"2. { 'max' if is_highest else 'min' }() -> {format_number(best_val)}{unit_str}",
                        f"3. {target_cat_col} -> {best_cat}"
                    ]
                },
                "data_used": data_used_rows,
                "chart_data": {
                    "type": "bar",
                    "title": f"{target_num_col} by {target_cat_col}",
                    "labels": labels,
                    "values": values,
                    "highlight_index": highlight_idx,
                    "x_label": str(target_cat_col),
                    "y_label": str(target_num_col)
                }
            }

    # =========================================================================
    # CASE F: OPERATION 5 - STANDALONE MIN / MAX (NO SPECIFIC CATEGORY/FILTER)
    # =========================================================================
    if is_highest or is_lowest:
        clean_series = pd.to_numeric(df[target_num_col], errors="coerce").dropna()
        val = clean_series.max() if is_highest else clean_series.min()
        adj = "highest" if is_highest else "lowest"

        # Find row context where this value occurred
        idx = clean_series.idxmax() if is_highest else clean_series.idxmin()
        match_row = df.loc[idx]

        context_info = []
        for c in categorical_cols:
            if c in match_row and pd.notna(match_row[c]):
                context_info.append(f"{match_row[c]}")
        context_str = f" (in {', '.join(context_info[:2])})" if context_info else ""

        answer_text = f"The {adj} {metric_label.lower()} recorded is {format_number(val)}{unit_str}{context_str}."

        return {
            "success": True,
            "question": question,
            "answer": answer_text,
            "operation": f"standalone_{adj}",
            "calculation_details": {
                "method": f"Find {adj} value in {target_num_col}",
                "formula": f"{'max' if is_highest else 'min'}() = {format_number(val)}{unit_str}",
                "steps": [
                    f"1. Evaluated all {len(clean_series):,} records in '{target_num_col}'",
                    f"2. Result: {format_number(val)}{unit_str}"
                ]
            },
            "data_used": [{"Metric": f"{adj.capitalize()} {target_num_col}", "Value": f"{format_number(val)}{unit_str}"}],
            "chart_data": None
        }

    # =========================================================================
    # CASE G: OPERATION 6 - AVERAGE / MEAN (OVERALL)
    # =========================================================================
    if is_avg:
        clean_series = pd.to_numeric(df[target_num_col], errors="coerce").dropna()
        avg_val = clean_series.mean()
        sum_val = clean_series.sum()
        count_val = clean_series.count()

        answer_text = f"The average {metric_label.lower()} across all {count_val:,} records is {format_number(avg_val)}{unit_str}."

        chart_data = None
        data_used = []
        if target_cat_col:
            grouped = df.groupby(target_cat_col)[target_num_col].mean().dropna()
            if not grouped.empty:
                chart_data = {
                    "type": "bar",
                    "title": f"Average {target_num_col} by {target_cat_col}",
                    "labels": [str(k) for k in grouped.index],
                    "values": [round(float(v), 2) for v in grouped.values],
                    "x_label": str(target_cat_col),
                    "y_label": f"Average {target_num_col}"
                }
                data_used = [{target_cat_col: str(k), f"Avg {target_num_col}": format_number(v)} for k, v in grouped.items()]

        if not data_used:
            data_used = [
                {"Metric": "Sum of Values", "Value": f"{format_number(sum_val)}{unit_str}"},
                {"Metric": "Total Records Counted", "Value": f"{count_val:,}"},
                {"Metric": "Average", "Value": f"{format_number(avg_val)}{unit_str}"}
            ]

        return {
            "success": True,
            "question": question,
            "answer": answer_text,
            "operation": "average",
            "calculation_details": {
                "method": f"Calculate arithmetic mean of {target_num_col}",
                "formula": f"Average = {format_number(sum_val)} / {count_val:,} = {format_number(avg_val)}{unit_str}",
                "steps": [
                    f"1. Sum of {metric_label}: {format_number(sum_val)}{unit_str}",
                    f"2. Total count of records: {count_val:,}",
                    f"3. Mean = {format_number(sum_val)} / {count_val:,} = {format_number(avg_val)}{unit_str}"
                ]
            },
            "data_used": data_used,
            "chart_data": chart_data
        }

    # =========================================================================
    # CASE H: OPERATION 7 - SUM / TOTAL (OVERALL)
    # =========================================================================
    if is_sum:
        clean_series = pd.to_numeric(df[target_num_col], errors="coerce").dropna()
        sum_val = clean_series.sum()
        count_val = clean_series.count()

        answer_text = f"The total {metric_label.lower()} across all {count_val:,} records is {format_number(sum_val)}{unit_str}."

        chart_data = None
        data_used = []
        if target_cat_col:
            grouped = df.groupby(target_cat_col)[target_num_col].sum().dropna()
            if not grouped.empty:
                chart_data = {
                    "type": "bar",
                    "title": f"Total {target_num_col} by {target_cat_col}",
                    "labels": [str(k) for k in grouped.index],
                    "values": [round(float(v), 2) for v in grouped.values],
                    "x_label": str(target_cat_col),
                    "y_label": f"Total {target_num_col}"
                }
                data_used = [{target_cat_col: str(k), f"Total {target_num_col}": format_number(v)} for k, v in grouped.items()]

        if not data_used:
            data_used = [
                {"Metric": "Total Sum", "Value": f"{format_number(sum_val)}{unit_str}"},
                {"Metric": "Records Counted", "Value": f"{count_val:,}"}
            ]

        return {
            "success": True,
            "question": question,
            "answer": answer_text,
            "operation": "sum",
            "calculation_details": {
                "method": f"Sum all values in {target_num_col}",
                "formula": f"sum({target_num_col}) = {format_number(sum_val)}{unit_str}",
                "steps": [
                    f"1. Summed {count_val:,} valid records in '{target_num_col}'",
                    f"2. Total = {format_number(sum_val)}{unit_str}"
                ]
            },
            "data_used": data_used,
            "chart_data": chart_data
        }

    # Fallback
    return {
        "success": False,
        "question": question,
        "answer": "I could not determine the specific calculation for your question. You can ask for averages, totals, rankings (e.g. 'which department consumes the most electricity'), comparisons, or time trends.",
        "operation": "unsupported",
        "error": "Question operation not recognized."
    }


# =============================================================================
# DATE-AWARE QUESTIONS (e.g. "last month", "this week", "last quarter")
# Resolves a relative period against today's date, filters the dataset by its
# date column, then delegates all calculations to the existing engine above.
# =============================================================================

_RELATIVE_PERIOD_RE = re.compile(
    r"\b(?:"
    r"(?P<rel>this|current|last|previous)\s+(?P<unit>week|month|quarter|year)"
    r"|(?P<single>today|yesterday)"
    r"|(?:last|past)\s+(?P<n>\d+)\s+(?P<nunit>days|weeks|months)"
    r")\b",
    re.IGNORECASE,
)


def find_relative_period(question: str):
    """Returns the regex match for a relative period phrase in the question, or None."""
    return _RELATIVE_PERIOD_RE.search(question)


def resolve_relative_period(match, today: datetime.date):
    """
    Converts a matched phrase into (start_date, end_date_inclusive, label) relative to `today`.
    Weeks run Monday-Sunday; quarters are calendar quarters (Jan-Mar, Apr-Jun, ...).
    """
    phrase = " ".join(match.group(0).lower().split())

    if match.group("single"):
        day = today if match.group("single").lower() == "today" else today - datetime.timedelta(days=1)
        return day, day, phrase

    if match.group("n"):
        n = int(match.group("n"))
        unit = match.group("nunit").lower()
        if unit == "days":
            start = today - datetime.timedelta(days=n - 1)
        elif unit == "weeks":
            start = today - datetime.timedelta(weeks=n) + datetime.timedelta(days=1)
        else:
            start = (pd.Timestamp(today) - pd.DateOffset(months=n) + pd.Timedelta(days=1)).date()
        return start, today, phrase

    unit = match.group("unit").lower()
    previous = match.group("rel").lower() in ("last", "previous")

    if unit == "week":
        start = today - datetime.timedelta(days=today.weekday())
        if previous:
            start -= datetime.timedelta(days=7)
        return start, start + datetime.timedelta(days=6), phrase

    if unit == "month":
        start = today.replace(day=1)
        if previous:
            start = (start - datetime.timedelta(days=1)).replace(day=1)
        end = (pd.Timestamp(start) + pd.offsets.MonthEnd(0)).date()
        return start, end, phrase

    if unit == "quarter":
        q_start_month = 3 * ((today.month - 1) // 3) + 1
        start = datetime.date(today.year, q_start_month, 1)
        if previous:
            start = (pd.Timestamp(start) - pd.DateOffset(months=3)).date()
        end = (pd.Timestamp(start) + pd.DateOffset(months=3) - pd.Timedelta(days=1)).date()
        return start, end, phrase

    # year
    year = today.year - 1 if previous else today.year
    return datetime.date(year, 1, 1), datetime.date(year, 12, 31), phrase


def analyze_question(df: pd.DataFrame, question: str, today: datetime.date = None) -> dict:
    """
    Public entry point. If the question contains a relative period ("last month",
    "this week", "last quarter", ...), filter the dataset to that period using the
    dataset's date column and run the normal analysis on the filtered rows.
    Questions without a relative period behave exactly as before.
    `today` defaults to the current date and can be overridden for testing.
    """
    if df is None or df.empty or not isinstance(question, str):
        return _analyze_question_core(df, question)

    match = find_relative_period(question)
    if not match:
        return _analyze_question_core(df, question)

    today = today or datetime.date.today()
    start, end, phrase = resolve_relative_period(match, today)
    period_text = f"{phrase} ({start.isoformat()} to {end.isoformat()})" if start != end else f"{phrase} ({start.isoformat()})"

    _, _, date_cols = identify_columns(df)
    if not date_cols:
        return {
            "success": False,
            "question": question,
            "answer": (
                f"Your question refers to \"{phrase}\", but the dataset has no recognizable date column, "
                f"so I can't filter by time. Available columns: {', '.join(str(c) for c in df.columns)}"
            ),
            "operation": "date_filter_no_column",
            "error": "No date column found for date-based filtering."
        }

    date_col = date_cols[0]
    try:
        parsed = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
    except (TypeError, ValueError):
        parsed = pd.to_datetime(df[date_col], errors="coerce")
    valid = parsed.dropna()
    if valid.empty:
        return {
            "success": False,
            "question": question,
            "answer": f"The column '{date_col}' does not contain any valid dates, so I can't filter by {phrase}.",
            "operation": "date_filter_no_column",
            "error": f"No parseable dates in '{date_col}'."
        }

    mask = (parsed >= pd.Timestamp(start)) & (parsed < pd.Timestamp(end) + pd.Timedelta(days=1))
    filtered_df = df[mask.fillna(False)]

    if filtered_df.empty:
        return {
            "success": False,
            "question": question,
            "answer": (
                f"No records found for {period_text}. Today is {today.isoformat()}, and the dataset's "
                f"'{date_col}' column covers {valid.min().date().isoformat()} to {valid.max().date().isoformat()}."
            ),
            "operation": "date_filter_no_records",
            "error": f"No rows in '{date_col}' fall within {start.isoformat()} to {end.isoformat()}."
        }

    # Remove the period phrase so the existing engine only sees the remaining question.
    cleaned = (question[:match.start()] + " " + question[match.end():])
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([?.!,])", r"\1", cleaned)

    result = _analyze_question_core(filtered_df, cleaned)
    result["question"] = question

    note = f"Date filter: {period_text} on '{date_col}' - {len(filtered_df):,} of {len(df):,} records."
    if result.get("success"):
        result["answer"] = f"{result['answer']} [Period: {period_text}]"
        details = result.get("calculation_details")
        if isinstance(details, dict):
            details["steps"] = [note] + list(details.get("steps", []))
    result["period"] = {
        "phrase": phrase,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "date_column": str(date_col),
        "records_in_period": int(len(filtered_df)),
    }
    return result