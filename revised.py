# =========================================================
#  Smart Data Preprocessing Studio (CSV + Excel + Target-safe)
# ---------------------------------------------------------
# Author: Shreyas 
# Description: Interactive, industry-ready preprocessing UI
# =========================================================

!pip install gradio pandas scikit-learn ydata-profiling openpyxl -q

import gradio as gr
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.feature_selection import VarianceThreshold
from ydata_profiling import ProfileReport

# =========================================================
#  Helper Functions
# =========================================================

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
        return df.fillna(method='ffill')
    elif method == "bfill":
        return df.fillna(method='bfill')
    elif method == "drop":
        return df.dropna()
    else:
        return df


def encode_features(df, encoding_type):
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    if not len(cat_cols): return df

    if encoding_type == "Label":
        le = LabelEncoder()
        for c in cat_cols:
            df[c] = le.fit_transform(df[c].astype(str))
    elif encoding_type == "One-Hot":
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    elif encoding_type == "Ordinal":
        oe = OrdinalEncoder()
        df[cat_cols] = oe.fit_transform(df[cat_cols].astype(str))
    return df


def scale_features(df, scaler_type):
    num_cols = df.select_dtypes(include=[np.number]).columns
    if not len(num_cols): return df

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
        return df[mask]
    elif method == "EllipticEnvelope":
        ee = EllipticEnvelope(contamination=0.05)
        try:
            mask = ee.fit_predict(num_cols) == 1
            return df[mask]
        except:
            return df
    return df


def feature_selection(df, threshold):
    sel = VarianceThreshold(threshold=threshold)
    num_cols = df.select_dtypes(include=[np.number])
    if not num_cols.empty:
        reduced = sel.fit_transform(num_cols)
        reduced_df = pd.DataFrame(reduced, columns=num_cols.columns[sel.get_support()])
        df = df.drop(columns=num_cols.columns)
        df = pd.concat([df, reduced_df], axis=1)
    return df


def create_profile(df, filename="profile_report.html"):
    profile = ProfileReport(df, title="Data Profiling Report", minimal=True)
    profile.to_file(filename)
    return filename


# =========================================================
# ⚙️ Main Pipeline
# =========================================================

def preprocess_pipeline(file, target_col, missing_method, fill_const, encoding_type,
                        scaler_type, outlier_method, var_thresh, do_profile):

    # --- Load File ---
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file.name)
        elif file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file.name)
        else:
            return " Unsupported file format. Upload CSV or Excel.", None, None
    except Exception as e:
        return f" Failed to read file: {e}", None, None

    original_shape = df.shape
    target = None

    # --- Target Protection ---
    if target_col and target_col in df.columns:
        target = df[target_col]
        df = df.drop(columns=[target_col])

    # --- Apply Steps ---
    config = {
        "target_column": target_col,
        "missing_method": missing_method,
        "fill_value": fill_const,
        "encoding": encoding_type,
        "scaler": scaler_type,
        "outlier_method": outlier_method,
        "variance_threshold": var_thresh,
        "profile_report": do_profile
    }

    df = handle_missing(df, missing_method, fill_const)
    df = encode_features(df, encoding_type)
    df = handle_outliers(df, outlier_method)
    df = feature_selection(df, var_thresh)
    df = scale_features(df, scaler_type)

    # --- Reattach Target ---
    if target is not None:
        df[target_col] = target.values

    # --- Save Outputs ---
    processed_name = "processed_dataset.csv"
    df.to_csv(processed_name, index=False)

    with open("job_config.json", "w") as f:
        json.dump(config, f, indent=4)

    report_path = None
    if do_profile:
        report_path = create_profile(df)

    summary = (
        f" Preprocessing Complete!\n"
        f"Original Shape: {original_shape}\n"
        f"New Shape: {df.shape}\n"
        f"Target Column: {target_col if target_col else 'None'}\n"
        f"Remaining Nulls: {df.isnull().sum().sum()}\n"
    )

    return summary, processed_name, report_path


# =========================================================
#  Gradio Interface
# =========================================================

def app():
    with gr.Blocks(theme="soft", title="Smart Data Preprocessing Studio") as demo:
        gr.Markdown("#  Smart Data Preprocessing Studio")
        gr.Markdown("### Upload, clean, encode, scale & profile your dataset — with target column safety!")

        file_input = gr.File(label="Upload CSV or Excel File", file_types=[".csv", ".xlsx", ".xls"])
        target_col = gr.Textbox(label=" Target Column (optional)", placeholder="e.g. target or label")

        with gr.Row():
            missing_method = gr.Dropdown(
                ["mean", "median", "most_frequent", "constant", "ffill", "bfill", "drop"],
                label="Missing Value Strategy",
                value="mean"
            )
            fill_const = gr.Textbox(label="Constant Fill Value (if selected)", value="")

        with gr.Row():
            encoding_type = gr.Dropdown(
                ["Label", "One-Hot", "Ordinal"],
                label="Encoding Type",
                value="Label"
            )
            scaler_type = gr.Dropdown(
                ["Standard", "MinMax", "Robust", "None"],
                label="Scaler Type",
                value="Standard"
            )

        with gr.Row():
            outlier_method = gr.Dropdown(
                ["None", "IQR", "EllipticEnvelope"],
                label="Outlier Handling",
                value="None"
            )
            var_thresh = gr.Slider(0.0, 0.2, step=0.01, label="Variance Threshold", value=0.0)

        do_profile = gr.Checkbox(label="Generate Profiling Report (ydata_profiling)", value=False)

        btn = gr.Button(" Run Preprocessing")
        summary = gr.Textbox(label=" Summary")
        processed_file = gr.File(label=" Download Processed CSV")
        report_file = gr.File(label=" Profiling Report (if generated)")

        btn.click(
            fn=preprocess_pipeline,
            inputs=[file_input, target_col, missing_method, fill_const, encoding_type,
                    scaler_type, outlier_method, var_thresh, do_profile],
            outputs=[summary, processed_file, report_file]
        )

    return demo


# =========================================================
# Launch in Colab
# =========================================================
demo = app()
demo.launch(share=False, debug=False, inline=True)

