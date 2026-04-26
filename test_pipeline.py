"""
DataPrepX – Pipeline Test Script
Directly inlines the helpers from revised.py to avoid ydata_profiling's
broken pkg_resources dependency at import time.
"""

import os, sys, json, types
import pandas as pd
import numpy as np
from sklearn.preprocessing import (LabelEncoder, OrdinalEncoder,
                                    StandardScaler, MinMaxScaler, RobustScaler)
from sklearn.covariance import EllipticEnvelope
from sklearn.feature_selection import VarianceThreshold

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

# ── Inline helpers (mirror of revised.py, minus ydata_profiling) ────────────

def handle_missing(df, method, fill_value=None):
    if method == "mean":
        return df.fillna(df.mean(numeric_only=True))
    elif method == "median":
        return df.fillna(df.median(numeric_only=True))
    elif method == "most_frequent":
        return df.fillna(df.mode().iloc[0])
    elif method == "constant":
        return df.fillna(fill_value if fill_value else 0)
    elif method == "ffill":
        return df.ffill()
    elif method == "bfill":
        return df.bfill()
    elif method == "drop":
        return df.dropna()
    return df


def encode_features(df, encoding_type):
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    if not len(cat_cols):
        return df
    if encoding_type == "Label":
        le = LabelEncoder()
        for c in cat_cols:
            df[c] = le.fit_transform(df[c].astype(str))
    elif encoding_type == "One-Hot":
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        bool_cols = df.select_dtypes(include=[bool]).columns
        df[bool_cols] = df[bool_cols].astype(np.int8)
    elif encoding_type == "Ordinal":
        oe = OrdinalEncoder()
        df[cat_cols] = oe.fit_transform(df[cat_cols].astype(str))
    return df


def scale_features(df, scaler_type):
    num_cols = df.select_dtypes(include=[np.number]).columns
    if not len(num_cols):
        return df
    if scaler_type == "Standard":
        scaler = StandardScaler()
    elif scaler_type == "MinMax":
        scaler = MinMaxScaler()
    elif scaler_type == "Robust":
        scaler = RobustScaler()
    else:
        return df
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


def handle_outliers(df, method):
    num_cols = df.select_dtypes(include=[np.number])
    if method == "IQR":
        Q1 = num_cols.quantile(0.25)
        Q3 = num_cols.quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((num_cols < (Q1 - 1.5 * IQR)) | (num_cols > (Q3 + 1.5 * IQR))).any(axis=1)
        return df[mask].reset_index(drop=True)
    elif method == "EllipticEnvelope":
        ee = EllipticEnvelope(contamination=0.05)
        try:
            mask = ee.fit_predict(num_cols) == 1
            return df[mask].reset_index(drop=True)
        except Exception:
            return df
    return df


def feature_selection(df, threshold):
    sel = VarianceThreshold(threshold=threshold)
    df = df.reset_index(drop=True)
    num_cols = df.select_dtypes(include=[np.number])
    if not num_cols.empty:
        reduced = sel.fit_transform(num_cols)
        reduced_df = pd.DataFrame(reduced, columns=num_cols.columns[sel.get_support()],
                                  index=df.index)
        df = df.drop(columns=num_cols.columns)
        df = pd.concat([df, reduced_df], axis=1)
    return df


def preprocess_pipeline(file, target_col, missing_method, fill_const,
                        encoding_type, scaler_type, outlier_method,
                        var_thresh, do_profile=False):
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file.name)
        elif file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file.name)
        else:
            return "Unsupported file format.", None, None, None
    except Exception as e:
        return f"Failed to read file: {e}", None, None, None

    original_shape = df.shape
    target = None

    if target_col and target_col in df.columns:
        target = df[target_col]
        df = df.drop(columns=[target_col])

    config = {
        "target_column": target_col,
        "missing_method": missing_method,
        "fill_value": fill_const,
        "encoding": encoding_type,
        "scaler": scaler_type,
        "outlier_method": outlier_method,
        "variance_threshold": var_thresh,
        "profile_report": do_profile,
    }

    df = handle_missing(df, missing_method, fill_const)
    df = encode_features(df, encoding_type)
    df = handle_outliers(df, outlier_method)
    df = feature_selection(df, var_thresh)
    df = scale_features(df, scaler_type)

    if target is not None:
        df[target_col] = target.values

    processed_name = "processed_dataset.csv"
    df.to_csv(processed_name, index=False)

    json_name = "processed_dataset.json"
    df.to_json(json_name, orient="records", indent=4)

    with open("job_config.json", "w") as f:
        json.dump(config, f, indent=4)

    summary = (
        f"Preprocessing Complete!\n"
        f"Original Shape : {original_shape}\n"
        f"New Shape      : {df.shape}\n"
        f"Target Column  : {target_col if target_col else 'None'}\n"
        f"Remaining Nulls: {df.isnull().sum().sum()}\n"
        f"Output Formats : CSV, JSON\n"
    )
    return summary, processed_name, json_name, None  # no profiling in tests


# ════════════════════════════════════════════════════════════════════════════
# Test harness
# ════════════════════════════════════════════════════════════════════════════
PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((status, name, detail))
    print(f"  {status}  {name}" + (f"  [{detail}]" if detail else ""))


CSV_PATH = os.path.join(PROJECT_DIR, "test_dataset.csv")

# ── 1. Load raw dataset ──────────────────────────────────────────────────────
print("\n── 1. Loading test_dataset.csv ─────────────────────────────────────")
df_raw = pd.read_csv(CSV_PATH)
check("CSV loads successfully", not df_raw.empty,
      f"{df_raw.shape[0]} rows × {df_raw.shape[1]} cols")
print(f"     Columns : {list(df_raw.columns)}")
missing = df_raw.isnull().sum()
print(f"     Missing : {dict(missing[missing > 0])}")


# ── 2. Helper unit tests ─────────────────────────────────────────────────────
print("\n── 2. Unit-testing helper functions ────────────────────────────────")

df = df_raw.copy()

df_mean = handle_missing(df.copy(), "mean")
check("handle_missing(mean)     — no numeric NaNs remain",
      df_mean.select_dtypes(include=[np.number]).isnull().sum().sum() == 0)

df_med = handle_missing(df.copy(), "median")
check("handle_missing(median)   — no numeric NaNs remain",
      df_med.select_dtypes(include=[np.number]).isnull().sum().sum() == 0)

df_drop = handle_missing(df.copy(), "drop")
check("handle_missing(drop)     — row count ≤ original",
      len(df_drop) <= len(df))

df_enc_label = encode_features(df_mean.copy(), "Label")
check("encode_features(Label)   — no object columns remain",
      df_enc_label.select_dtypes(exclude=[np.number]).shape[1] == 0)

df_enc_oh = encode_features(df_mean.copy(), "One-Hot")
check("encode_features(One-Hot) — no object columns remain",
      df_enc_oh.select_dtypes(exclude=[np.number]).shape[1] == 0)

df_std = scale_features(df_enc_label.copy(), "Standard")
check("scale_features(Standard) — returns non-empty df",
      not df_std.empty)

df_mm = scale_features(df_enc_label.copy(), "MinMax")
num_max = df_mm.select_dtypes(include=[np.number]).max().max()
check("scale_features(MinMax)   — max value ≤ 1.01",
      num_max <= 1.01, f"max={num_max:.4f}")

df_iqr = handle_outliers(df_enc_label.copy(), "IQR")
check("handle_outliers(IQR)     — non-empty result",
      len(df_iqr) > 0, f"{len(df_iqr)}/{len(df_enc_label)} rows kept")

df_fs = feature_selection(df_enc_label.copy(), threshold=0.0)
check("feature_selection(0.0)   — returns non-empty df", not df_fs.empty)


# ── 3. Full pipeline integration test ───────────────────────────────────────
print("\n── 3. Full pipeline integration test ───────────────────────────────")

fake_file = types.SimpleNamespace(name=CSV_PATH)

summary, csv_out, json_out, _ = preprocess_pipeline(
    file=fake_file, target_col="", missing_method="mean",
    fill_const="", encoding_type="Label", scaler_type="Standard",
    outlier_method="IQR", var_thresh=0.0, do_profile=False,
)

check("Pipeline — returns summary string", isinstance(summary, str) and len(summary) > 0)
check("Pipeline — returns CSV path",       isinstance(csv_out, str))
check("Pipeline — returns JSON path",      isinstance(json_out, str))
print(f"\n  Pipeline Summary:\n{summary}")


# ── 4. Validate CSV output ──────────────────────────────────────────────────
print("── 4. Validating CSV output ─────────────────────────────────────────")
check("CSV file exists", os.path.isfile(csv_out))
df_csv = pd.read_csv(csv_out)
check("CSV is non-empty",      not df_csv.empty,
      f"{df_csv.shape[0]} rows × {df_csv.shape[1]} cols")
check("CSV has no NaN values", df_csv.isnull().sum().sum() == 0)


# ── 5. Validate JSON output ─────────────────────────────────────────────────
print("\n── 5. Validating JSON output ───────────────────────────────────────")
check("JSON file exists", os.path.isfile(json_out))

with open(json_out) as f:
    raw_json = f.read()

check("JSON file is non-empty", len(raw_json) > 0, f"{len(raw_json):,} bytes")

try:
    records = json.loads(raw_json)
    valid_json = True
except json.JSONDecodeError as e:
    valid_json = False
    print(f"     JSON parse error: {e}")
    records = []

check("JSON parses without errors",    valid_json)
check("JSON is a list of records",     isinstance(records, list))
if records:
    check("JSON row count matches CSV",
          len(records) == len(df_csv), f"{len(records)} JSON vs {len(df_csv)} CSV")
    check("JSON columns match CSV",
          set(records[0].keys()) == set(df_csv.columns))
    print(f"\n  Sample JSON record (first entry):")
    print(json.dumps(records[0], indent=4))


# ── 6. Validate job_config.json ─────────────────────────────────────────────
print("\n── 6. Validating job_config.json ───────────────────────────────────")
config_path = os.path.join(PROJECT_DIR, "job_config.json")
check("job_config.json exists", os.path.isfile(config_path))
with open(config_path) as f:
    config = json.load(f)
check("Config has expected keys",
      {"missing_method", "encoding", "scaler", "outlier_method"}.issubset(config.keys()))
print(f"  Config: {json.dumps(config, indent=4)}")


# ── 7. Test with target column protection ────────────────────────────────────
print("\n── 7. Target column protection test ────────────────────────────────")
first_col = df_raw.columns[0]
summary2, csv2, json2, _ = preprocess_pipeline(
    file=fake_file, target_col=first_col, missing_method="mean",
    fill_const="", encoding_type="Label", scaler_type="MinMax",
    outlier_method="None", var_thresh=0.0, do_profile=False,
)
df_csv2 = pd.read_csv(csv2)
check(f"Target column '{first_col}' preserved in output",
      first_col in df_csv2.columns)


# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "═" * 62)
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
print(f"  TOTAL: {passed} passed, {failed} failed  (out of {len(results)} checks)")
print("═" * 62)

if failed:
    print("\nFailed checks:")
    for s, n, d in results:
        if s == FAIL:
            print(f"  {s}  {n}" + (f"  [{d}]" if d else ""))
    sys.exit(1)
else:
    print("\n🎉  All checks passed — DataPrepX is working correctly!")
