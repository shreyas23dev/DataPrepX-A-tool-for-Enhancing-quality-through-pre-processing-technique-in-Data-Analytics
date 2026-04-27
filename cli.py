#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║          DataPrepX  —  Interactive Terminal Setup            ║
║  Smart Data Preprocessing with full CSV + JSON output        ║
╚══════════════════════════════════════════════════════════════╝

Run:
    python cli.py
    python cli.py --file path/to/data.csv   # skip file prompt
"""

import os
import re
import sys
import json
import time
import argparse
import requests

import numpy as np
import pandas as pd
import questionary
from questionary import Style
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich import print as rprint

from sklearn.preprocessing import (
    LabelEncoder, OrdinalEncoder,
    StandardScaler, MinMaxScaler, RobustScaler,
)
from sklearn.covariance import EllipticEnvelope
from sklearn.feature_selection import VarianceThreshold

# ─────────────────────────────────────────────────────────────
#  Console & Style setup
# ─────────────────────────────────────────────────────────────
console = Console()

Q_STYLE = Style([
    ("qmark",        "fg:#00d7af bold"),
    ("question",     "bold"),
    ("answer",       "fg:#00d7af bold"),
    ("pointer",      "fg:#00d7af bold"),
    ("highlighted",  "fg:#00d7af bold"),
    ("selected",     "fg:#00d7af"),
    ("separator",    "fg:#6c6c6c"),
    ("instruction",  "fg:#858585 italic"),
])

# ─────────────────────────────────────────────────────────────
#  Preprocessing helpers  (fixed versions from test phase)
# ─────────────────────────────────────────────────────────────

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
    scalers = {
        "Standard": StandardScaler(),
        "MinMax":   MinMaxScaler(),
        "Robust":   RobustScaler(),
    }
    if scaler_type not in scalers:
        return df
    df[num_cols] = scalers[scaler_type].fit_transform(df[num_cols])
    return df


def handle_outliers(df, method):
    num_cols = df.select_dtypes(include=[np.number])
    if method == "IQR":
        Q1  = num_cols.quantile(0.25)
        Q3  = num_cols.quantile(0.75)
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
    sel     = VarianceThreshold(threshold=threshold)
    df      = df.reset_index(drop=True)
    num_cols = df.select_dtypes(include=[np.number])
    if not num_cols.empty:
        reduced    = sel.fit_transform(num_cols)
        reduced_df = pd.DataFrame(
            reduced,
            columns=num_cols.columns[sel.get_support()],
            index=df.index,
        )
        df = df.drop(columns=num_cols.columns)
        df = pd.concat([df, reduced_df], axis=1)
    return df


# ─────────────────────────────────────────────────────────────
#  Banner
# ─────────────────────────────────────────────────────────────

def print_banner():
    console.print()
    console.print(Panel.fit(
        "[bold cyan]DataPrepX[/bold cyan]  [dim]—[/dim]  "
        "[bold white]Smart Data Preprocessing Studio[/bold white]\n"
        "[dim]Interactive Terminal  ·  CSV · JSON · XLSX output[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))
    console.print()


# ─────────────────────────────────────────────────────────────
#  Interactive Cell Editor
# ─────────────────────────────────────────────────────────────

DEFAULT_OLLAMA_MODEL = "llama3.2:max_context"
OLLAMA_BASE_URL      = "http://localhost:11434"


def select_and_edit_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Interactive cell-level editor: select column + row index, inspect context, edit."""
    console.print(Rule("[bold cyan]Cell Editor[/bold cyan]"))
    console.print("[dim]Select individual cells to inspect and edit before preprocessing.[/dim]\n")

    col_names = list(df.columns)

    while True:
        action = questionary.select(
            "Cell Editor — what would you like to do?",
            choices=[
                questionary.Choice("Edit a cell     — select column + row", "edit"),
                questionary.Choice("View a region   — show rows around an index", "view"),
                questionary.Choice("Done editing    — continue to config", "done"),
            ],
            style=Q_STYLE,
        ).ask()

        if action == "done" or action is None:
            break

        # ── Column selection ──────────────────────────────────
        col = questionary.autocomplete(
            "Column name:",
            choices=col_names,
            style=Q_STYLE,
            validate=lambda v: v in col_names or f"Choose one of: {', '.join(col_names[:8])}…",
        ).ask()
        if col is None:
            break

        # ── Row index selection ───────────────────────────────
        row_str = questionary.text(
            f"Row index (0 – {len(df) - 1}):",
            style=Q_STYLE,
            validate=lambda v: (
                True if v.isdigit() and 0 <= int(v) < len(df)
                else f"Enter an integer between 0 and {len(df) - 1}"
            ),
        ).ask()
        if row_str is None:
            break
        row_idx = int(row_str)

        # ── Context preview ───────────────────────────────────
        lo  = max(0, row_idx - 2)
        hi  = min(len(df), row_idx + 3)
        ctx = df.iloc[lo:hi][[col]].copy()
        ctx.index.name = "row"

        ctx_table = Table(
            title=f"Column: [bold cyan]{col}[/bold cyan]  ·  rows {lo}–{hi - 1}",
            box=box.ROUNDED, border_style="dim cyan", header_style="bold cyan",
        )
        ctx_table.add_column("Row",   style="dim",        width=8)
        ctx_table.add_column("Value", style="bold white", no_wrap=True)
        for r, (ridx, row) in enumerate(ctx.iterrows()):
            val_str = str(row[col])
            style   = "bold yellow" if ridx == row_idx else ""
            ctx_table.add_row(
                f"[{style}]{'→ ' if ridx == row_idx else '  '}{ridx}[/{style}]",
                f"[{style}]{val_str[:80]}[/{style}]",
            )
        console.print(ctx_table)
        console.print()

        current_val = df.at[row_idx, col]
        console.print(f"  [dim]Current value:[/dim]  [bold white]{current_val!r}[/bold white]  "
                      f"[dim](dtype: {df[col].dtype})[/dim]")
        console.print()

        if action == "view":
            continue

        # ── Edit action ───────────────────────────────────────
        edit_action = questionary.select(
            "What would you like to do with this cell?",
            choices=[
                questionary.Choice("Overwrite   — enter a new value",          "overwrite"),
                questionary.Choice("Set to NaN  — mark as missing",            "nan"),
                questionary.Choice("Skip        — leave unchanged",            "skip"),
            ],
            style=Q_STYLE,
        ).ask()

        if edit_action == "overwrite":
            new_val_str = questionary.text(
                "New value:",
                default=str(current_val),
                style=Q_STYLE,
            ).ask()
            if new_val_str is not None:
                # Try to cast to the column's original dtype
                try:
                    if pd.api.types.is_integer_dtype(df[col]):
                        new_val = int(new_val_str)
                    elif pd.api.types.is_float_dtype(df[col]):
                        new_val = float(new_val_str)
                    else:
                        new_val = new_val_str
                    df.at[row_idx, col] = new_val
                    console.print(
                        f"  [bold green]✓[/bold green]  "
                        f"[cyan]{col}[/cyan][{row_idx}]  "
                        f"← [bold white]{new_val!r}[/bold white]\n"
                    )
                except (ValueError, TypeError) as exc:
                    console.print(f"  [bold red]✗[/bold red]  Could not cast value: {exc}\n")

        elif edit_action == "nan":
            df.at[row_idx, col] = np.nan
            console.print(
                f"  [bold green]✓[/bold green]  "
                f"[cyan]{col}[/cyan][{row_idx}] set to [yellow]NaN[/yellow]\n"
            )

    console.print(f"[bold green]✓[/bold green]  Cell editing complete.  "
                  f"[dim]DataFrame shape: {df.shape[0]:,} × {df.shape[1]}[/dim]\n")
    return df


# ─────────────────────────────────────────────────────────────
#  Ollama LLM Assistant helpers
# ─────────────────────────────────────────────────────────────

def _list_ollama_models() -> list[str]:
    """Return names of locally available Ollama models."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if r.ok:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


def build_data_snapshot(df: pd.DataFrame, max_rows: int = 5) -> str:
    """Compact text summary of df — injected as the LLM system context."""
    lines = [
        f"DataFrame shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n",
        "Column info (name | dtype | null_count | sample_value):",
    ]
    for col in df.columns:
        sample = df[col].dropna().iloc[0] if df[col].dropna().shape[0] else "—"
        lines.append(
            f"  {col!r:<30} {str(df[col].dtype):<12} "
            f"nulls={df[col].isnull().sum():<6} sample={str(sample)[:40]!r}"
        )

    lines.append(f"\nFirst {min(max_rows, len(df))} rows (truncated to 80 chars per cell):")
    try:
        preview = df.head(max_rows).to_string(max_colwidth=80)
        lines.append(preview)
    except Exception:
        lines.append("(could not render preview)")

    return "\n".join(lines)


# ─── Restricted eval whitelist ────────────────────────────────────────────────
_BLOCKED = re.compile(
    r"\b(os|sys|subprocess|open|exec|eval|compile|__import__|importlib"
    r"|socket|shutil|pathlib|glob|requests)\b"
)


def query_data_tool(df: pd.DataFrame, expression: str) -> str:
    """Safely evaluate a pandas expression against the live DataFrame."""
    if _BLOCKED.search(expression):
        return "[BLOCKED] Expression contains disallowed identifiers."
    try:
        # Provide `df` and common aliases in the eval namespace
        ns  = {"df": df, "pd": pd, "np": np}
        result = eval(expression, {"__builtins__": {}}, ns)  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"[ERROR] {type(exc).__name__}: {exc}"


# ─── Tool-call detector (JSON code block in model reply) ─────────────────────
_TOOL_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)


def _extract_tool_call(text: str) -> dict | None:
    """Look for a JSON tool-call block in the model's reply."""
    match = _TOOL_RE.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(1))
        if isinstance(obj, dict) and obj.get("name") == "query_data":
            return obj
    except json.JSONDecodeError:
        pass
    return None


def ollama_chat_loop(df: pd.DataFrame, stage_label: str = "Data") -> None:
    """Interactive LLM chat loop powered by a local Ollama model."""
    # ── Model selection ───────────────────────────────────────
    available = _list_ollama_models()
    if not available:
        console.print(
            "[bold red]✗[/bold red]  No Ollama models found. "
            "Make sure Ollama is running ([cyan]ollama serve[/cyan]).\n"
        )
        return

    default_model = (
        DEFAULT_OLLAMA_MODEL
        if DEFAULT_OLLAMA_MODEL in available
        else available[0]
    )

    console.print(Rule(f"[bold cyan]LLM Assistant — {stage_label}[/bold cyan]"))
    console.print(
        "[dim]The assistant can inspect your data with the "
        "[bold white]query_data[/bold white] tool.\n"
        "Type [bold white]exit[/bold white] or [bold white]quit[/bold white] to leave.[/dim]\n"
    )

    model = questionary.select(
        "Select Ollama model:",
        choices=available,
        default=default_model,
        style=Q_STYLE,
    ).ask()
    if model is None:
        return

    # ── System prompt ─────────────────────────────────────────
    snapshot  = build_data_snapshot(df)
    tool_spec = json.dumps({
        "name": "query_data",
        "description": (
            "Run a pandas expression on the live DataFrame `df` and return "
            "the result as a string. Use this to answer questions about the data. "
            "Always reference the DataFrame as `df`."
        ),
        "parameters": {
            "expression": "A valid pandas expression string, e.g. \"df['Age'].describe()\""
        },
    }, indent=2)

    system_prompt = (
        f"You are DataPrepX Assistant, an expert data analyst helping the user understand "
        f"and preprocess their dataset ({stage_label} stage).\n\n"
        f"=== DATASET SNAPSHOT ===\n{snapshot}\n\n"
        f"=== TOOL AVAILABLE ===\n"
        f"You have access to one tool:\n{tool_spec}\n\n"
        f"To call the tool, respond with ONLY a JSON code block like this:\n"
        f"```json\n{{\"name\": \"query_data\", \"parameters\": {{\"expression\": \"<expr>\"}}}}\n```\n"
        f"After receiving the tool result, incorporate it into a clear, concise answer. "
        f"Do not call the tool more than once per user message. "
        f"If the user's question can be answered from the snapshot alone, answer directly."
    )

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    console.print(f"  [bold green]●[/bold green]  Model: [cyan]{model}[/cyan]\n")

    # ── Chat loop ─────────────────────────────────────────────
    while True:
        try:
            user_input = questionary.text(
                "You:",
                style=Q_STYLE,
                multiline=False,
            ).ask()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input is None or user_input.strip().lower() in ("exit", "quit", ""):
            if user_input is None or user_input.strip().lower() in ("exit", "quit"):
                console.print("[dim]Leaving LLM assistant.[/dim]\n")
            break

        messages.append({"role": "user", "content": user_input.strip()})

        def _call_ollama(msgs: list[dict]) -> str:
            try:
                r = requests.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={"model": model, "messages": msgs, "stream": False},
                    timeout=120,
                )
                r.raise_for_status()
                return r.json()["message"]["content"]
            except requests.exceptions.ConnectionError:
                return "[ERROR] Cannot connect to Ollama. Is `ollama serve` running?"
            except Exception as exc:
                return f"[ERROR] {exc}"

        # ── First model call ──────────────────────────────────
        with console.status(f"[cyan]{model} is thinking…[/cyan]"):
            reply = _call_ollama(messages)

        # ── Tool-call handling ────────────────────────────────
        tool_call = _extract_tool_call(reply)
        if tool_call:
            expr = tool_call.get("parameters", {}).get("expression", "")
            console.print(
                f"  [dim]⚙ tool:[/dim] [yellow]query_data[/yellow]([cyan]{expr[:80]}[/cyan])"
            )
            with console.status("[cyan]Running query…[/cyan]"):
                tool_result = query_data_tool(df, expr)

            # Feed tool result back
            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "user",
                "content": f"[Tool result for query_data({expr!r})]:\n{tool_result}",
            })

            with console.status(f"[cyan]{model} is composing answer…[/cyan]"):
                reply = _call_ollama(messages)

        # ── Display response ──────────────────────────────────
        console.print()
        console.print(
            Panel(
                reply,
                title=f"[bold cyan]{model}[/bold cyan]",
                border_style="cyan",
                padding=(0, 2),
            )
        )
        console.print()

        messages.append({"role": "assistant", "content": reply})


# ─────────────────────────────────────────────────────────────
#  Dataset preview
# ─────────────────────────────────────────────────────────────

def preview_dataset(df: pd.DataFrame):
    console.print(Rule("[bold cyan]Dataset Preview[/bold cyan]"))

    # Shape + missing info
    meta = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    meta.add_column("key",   style="dim")
    meta.add_column("value", style="bold white")
    meta.add_row("Rows",     str(df.shape[0]))
    meta.add_row("Columns",  str(df.shape[1]))
    meta.add_row("Missing",  str(df.isnull().sum().sum()))
    console.print(meta)

    # Column info table
    col_table = Table(title="Columns", box=box.ROUNDED, border_style="dim cyan",
                      show_lines=False, header_style="bold cyan")
    col_table.add_column("#",       style="dim",         width=4)
    col_table.add_column("Name",    style="bold white",  no_wrap=True)
    col_table.add_column("Dtype",   style="yellow")
    col_table.add_column("Missing", style="red",         justify="right")
    col_table.add_column("Sample",  style="dim",         no_wrap=True)

    for i, col in enumerate(df.columns):
        sample = str(df[col].dropna().iloc[0]) if df[col].dropna().shape[0] else "—"
        col_table.add_row(
            str(i),
            col,
            str(df[col].dtype),
            str(df[col].isnull().sum()) if df[col].isnull().sum() else "0",
            sample[:30],
        )
    console.print(col_table)
    console.print()


# ─────────────────────────────────────────────────────────────
#  Interactive config wizard
# ─────────────────────────────────────────────────────────────

def ask_config(df: pd.DataFrame, file_path: str) -> dict:
    console.print(Rule("[bold cyan]Preprocessing Configuration[/bold cyan]"))
    console.print("[dim]Use arrow keys to select · Enter to confirm[/dim]\n")

    col_names = ["(none)"] + list(df.columns)

    # Target column
    target_col = questionary.select(
        "Target / label column to protect (will not be preprocessed):",
        choices=col_names,
        default="(none)",
        style=Q_STYLE,
    ).ask()
    if target_col == "(none)":
        target_col = ""

    console.print()

    # Missing value strategy
    missing_method = questionary.select(
        "Missing value strategy:",
        choices=[
            questionary.Choice("mean           — fill with column mean",        "mean"),
            questionary.Choice("median         — fill with column median",      "median"),
            questionary.Choice("most_frequent  — fill with most common value",  "most_frequent"),
            questionary.Choice("constant       — fill with a fixed value",      "constant"),
            questionary.Choice("ffill          — forward fill",                 "ffill"),
            questionary.Choice("bfill          — backward fill",                "bfill"),
            questionary.Choice("drop           — drop rows with missing values","drop"),
        ],
        style=Q_STYLE,
    ).ask()

    fill_const = ""
    if missing_method == "constant":
        fill_const = questionary.text(
            "Constant fill value:",
            default="0",
            style=Q_STYLE,
        ).ask()

    console.print()

    # Encoding
    encoding_type = questionary.select(
        "Categorical encoding:",
        choices=[
            questionary.Choice("Label    — integer label per category",     "Label"),
            questionary.Choice("One-Hot  — binary column per category",     "One-Hot"),
            questionary.Choice("Ordinal  — ordered integer encoding",       "Ordinal"),
            questionary.Choice("None     — skip encoding",                  "None"),
        ],
        style=Q_STYLE,
    ).ask()

    console.print()

    # Scaler
    scaler_type = questionary.select(
        "Feature scaling:",
        choices=[
            questionary.Choice("Standard  — zero mean, unit variance",    "Standard"),
            questionary.Choice("MinMax    — scale to [0, 1]",             "MinMax"),
            questionary.Choice("Robust    — median-based, outlier safe",  "Robust"),
            questionary.Choice("None      — skip scaling",                "None"),
        ],
        style=Q_STYLE,
    ).ask()

    console.print()

    # Outlier handling
    outlier_method = questionary.select(
        "Outlier handling:",
        choices=[
            questionary.Choice("None              — keep all rows",                    "None"),
            questionary.Choice("IQR               — remove rows outside 1.5×IQR",     "IQR"),
            questionary.Choice("EllipticEnvelope  — robust statistical detection",    "EllipticEnvelope"),
        ],
        style=Q_STYLE,
    ).ask()

    console.print()

    # Variance threshold
    var_thresh_str = questionary.select(
        "Feature selection — minimum variance threshold:",
        choices=[
            questionary.Choice("0.00  — keep all features",  "0.00"),
            questionary.Choice("0.01",                       "0.01"),
            questionary.Choice("0.05",                       "0.05"),
            questionary.Choice("0.10",                       "0.10"),
            questionary.Choice("0.20",                       "0.20"),
        ],
        style=Q_STYLE,
    ).ask()
    var_thresh = float(var_thresh_str)

    console.print()

    # Output formats
    output_formats = questionary.checkbox(
        "Output formats (space to toggle, enter to confirm):",
        choices=[
            questionary.Choice("CSV   — processed_<filename>.csv",  "csv",  checked=True),
            questionary.Choice("JSON  — processed_<filename>.json", "json", checked=True),
            questionary.Choice("XLSX  — processed_<filename>.xlsx", "xlsx", checked=True),
        ],
        style=Q_STYLE,
    ).ask()
    if not output_formats:
        output_formats = ["csv"]   # default fallback

    console.print()

    # Output directory
    out_dir = questionary.text(
        "Output directory (leave blank = same directory as input):",
        default="",
        style=Q_STYLE,
    ).ask()
    if not out_dir:
        out_dir = os.path.dirname(os.path.abspath(file_path))

    console.print()

    return dict(
        target_col     = target_col,
        missing_method = missing_method,
        fill_const     = fill_const,
        encoding_type  = encoding_type if encoding_type != "None" else None,
        scaler_type    = scaler_type   if scaler_type   != "None" else None,
        outlier_method = outlier_method,
        var_thresh     = var_thresh,
        output_formats = output_formats,
        out_dir        = out_dir,
    )


# ─────────────────────────────────────────────────────────────
#  Run pipeline with live progress
# ─────────────────────────────────────────────────────────────

def run_pipeline(df: pd.DataFrame, cfg: dict):
    steps = [
        ("Protecting target column",  None),
        ("Handling missing values",   lambda d: handle_missing(d, cfg["missing_method"], cfg["fill_const"])),
        ("Encoding categorical cols", lambda d: encode_features(d, cfg["encoding_type"]) if cfg["encoding_type"] else d),
        ("Removing outliers",         lambda d: handle_outliers(d, cfg["outlier_method"])),
        ("Feature selection",         lambda d: feature_selection(d, cfg["var_thresh"])),
        ("Scaling features",          lambda d: scale_features(d, cfg["scaler_type"]) if cfg["scaler_type"] else d),
    ]

    original_shape = df.shape
    target_series  = None

    console.print(Rule("[bold cyan]Running Pipeline[/bold cyan]"))

    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold white]{task.description}"),
        BarColumn(bar_width=30, style="cyan", complete_style="bold cyan"),
        TextColumn("[dim]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Starting…", total=len(steps))

        for label, fn in steps:
            progress.update(task, description=label)
            time.sleep(0.05)   # brief pause so each step is visible

            if label.startswith("Protecting"):
                if cfg["target_col"] and cfg["target_col"] in df.columns:
                    target_series = df[cfg["target_col"]].copy()
                    df = df.drop(columns=[cfg["target_col"]])
            elif fn is not None:
                df = fn(df)

            progress.advance(task)

    # Reattach target
    if target_series is not None:
        df[cfg["target_col"]] = target_series.values

    return df, original_shape


# ─────────────────────────────────────────────────────────────
#  Save outputs
# ─────────────────────────────────────────────────────────────

def save_outputs(df: pd.DataFrame, file_path: str, cfg: dict) -> dict:
    basename    = os.path.splitext(os.path.basename(file_path))[0]
    out_dir     = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    saved       = {}

    if "csv" in cfg["output_formats"]:
        csv_path = os.path.join(out_dir, f"processed_{basename}.csv")
        df.to_csv(csv_path, index=False)
        saved["csv"] = csv_path

    if "json" in cfg["output_formats"]:
        json_path = os.path.join(out_dir, f"processed_{basename}.json")
        df.to_json(json_path, orient="records", indent=4)
        saved["json"] = json_path

    if "xlsx" in cfg["output_formats"]:
        xlsx_path = os.path.join(out_dir, f"processed_{basename}.xlsx")
        df.to_excel(xlsx_path, index=False, engine="openpyxl")
        saved["xlsx"] = xlsx_path

    # Always save job config
    config_record = {
        "source_file":      file_path,
        "target_column":    cfg["target_col"],
        "missing_method":   cfg["missing_method"],
        "fill_value":       cfg["fill_const"],
        "encoding":         cfg["encoding_type"],
        "scaler":           cfg["scaler_type"],
        "outlier_method":   cfg["outlier_method"],
        "variance_threshold": cfg["var_thresh"],
        "output_formats":   cfg["output_formats"],
        "output_directory": out_dir,
    }
    config_path = os.path.join(out_dir, "job_config.json")
    with open(config_path, "w") as f:
        json.dump(config_record, f, indent=4)
    saved["config"] = config_path

    return saved


# ─────────────────────────────────────────────────────────────
#  Results summary
# ─────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, original_shape, cfg: dict, saved: dict):
    console.print()
    console.print(Rule("[bold green]Preprocessing Complete[/bold green]"))

    # Stats table
    stats = Table(box=box.ROUNDED, border_style="green", show_header=False, padding=(0, 2))
    stats.add_column("key",   style="dim",        min_width=22)
    stats.add_column("value", style="bold white")

    rows_removed = original_shape[0] - df.shape[0]
    cols_removed = original_shape[1] - df.shape[1] - (1 if cfg["target_col"] else 0)

    stats.add_row("Original shape",    f"{original_shape[0]:,} rows × {original_shape[1]} cols")
    stats.add_row("Processed shape",   f"[bold green]{df.shape[0]:,} rows × {df.shape[1]} cols[/bold green]")
    stats.add_row("Rows removed",      f"[yellow]{rows_removed:,}[/yellow]" if rows_removed else "[dim]0[/dim]")
    stats.add_row("Cols removed",      f"[yellow]{cols_removed:,}[/yellow]" if cols_removed > 0 else "[dim]0[/dim]")
    stats.add_row("Remaining nulls",   f"[red]{df.isnull().sum().sum()}[/red]" if df.isnull().sum().sum() else "[bold green]0 ✓[/bold green]")
    stats.add_row("Missing strategy",  cfg["missing_method"])
    stats.add_row("Encoding",          cfg["encoding_type"] or "—")
    stats.add_row("Scaler",            cfg["scaler_type"]   or "—")
    stats.add_row("Outlier method",    cfg["outlier_method"])
    stats.add_row("Variance threshold",str(cfg["var_thresh"]))
    stats.add_row("Target protected",  f"[bold cyan]{cfg['target_col']}[/bold cyan]" if cfg["target_col"] else "[dim]—[/dim]")

    console.print(stats)
    console.print()

    # Saved files table
    files_table = Table(title="Saved Files", box=box.ROUNDED,
                        border_style="cyan", header_style="bold cyan")
    files_table.add_column("Type",   style="bold white", width=10)
    files_table.add_column("Path",   style="cyan")
    files_table.add_column("Size",   style="dim", justify="right")

    for fmt, path in saved.items():
        size = os.path.getsize(path)
        size_str = f"{size / 1024:.1f} KB" if size < 1_048_576 else f"{size / 1_048_576:.1f} MB"
        files_table.add_row(fmt.upper(), path, size_str)

    console.print(files_table)
    console.print()

    console.print(Panel(
        "[bold green]✓[/bold green]  DataPrepX finished successfully!\n"
        "[dim]Your processed dataset is ready for modelling.[/dim]",
        border_style="green",
        padding=(0, 2),
    ))
    console.print()


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DataPrepX — Interactive Terminal Preprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--file", "-f",
        help="Path to input CSV or Excel file (skip file prompt)",
        default=None,
    )
    args = parser.parse_args()

    print_banner()

    # ── File selection ────────────────────────────────────────
    if args.file:
        file_path = args.file
    else:
        console.print(Rule("[bold cyan]File Selection[/bold cyan]"))
        file_path = questionary.path(
            "Path to your dataset (CSV or Excel):",
            style=Q_STYLE,
            only_directories=False,
        ).ask()

    if not file_path or not os.path.isfile(file_path):
        console.print(f"[bold red]✗[/bold red]  File not found: [cyan]{file_path}[/cyan]")
        sys.exit(1)

    # ── Load ─────────────────────────────────────────────────
    console.print()
    with console.status("[cyan]Loading dataset…[/cyan]"):
        try:
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
            elif file_path.endswith((".xlsx", ".xls")):
                df = pd.read_excel(file_path)
            else:
                console.print("[bold red]✗[/bold red]  Unsupported format. Use CSV or Excel.")
                sys.exit(1)
        except Exception as e:
            console.print(f"[bold red]✗[/bold red]  Could not read file: {e}")
            sys.exit(1)

    console.print(f"[bold green]✓[/bold green]  Loaded [bold white]{os.path.basename(file_path)}[/bold white]  "
                  f"[dim]({df.shape[0]:,} rows × {df.shape[1]} cols)[/dim]")
    console.print()

    # ── Preview ───────────────────────────────────────────────
    show_preview = questionary.confirm(
        "Show dataset preview before configuring?",
        default=True,
        style=Q_STYLE,
    ).ask()
    if show_preview:
        preview_dataset(df)

    # ── Cell editor ───────────────────────────────────────────
    want_edit = questionary.confirm(
        "Edit individual cells before preprocessing?",
        default=False,
        style=Q_STYLE,
    ).ask()
    if want_edit:
        df = select_and_edit_cells(df)

    # ── LLM assistant (raw data) ──────────────────────────────
    want_llm_raw = questionary.confirm(
        "Chat with the LLM assistant about this (raw) data?",
        default=False,
        style=Q_STYLE,
    ).ask()
    if want_llm_raw:
        ollama_chat_loop(df, stage_label="Raw Data")

    # ── Config wizard ─────────────────────────────────────────
    cfg = ask_config(df, file_path)

    # ── Confirm ───────────────────────────────────────────────
    console.print(Rule("[bold cyan]Confirm & Run[/bold cyan]"))
    confirm_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    confirm_table.add_column("step",   style="dim")
    confirm_table.add_column("choice", style="bold white")
    confirm_table.add_row("Missing strategy",   cfg["missing_method"])
    confirm_table.add_row("Encoding",           cfg["encoding_type"] or "—")
    confirm_table.add_row("Scaler",             cfg["scaler_type"]   or "—")
    confirm_table.add_row("Outlier handling",   cfg["outlier_method"])
    confirm_table.add_row("Variance threshold", str(cfg["var_thresh"]))
    confirm_table.add_row("Target column",      cfg["target_col"] or "—")
    confirm_table.add_row("Output formats",     ", ".join(cfg["output_formats"]).upper())
    confirm_table.add_row("Output directory",   cfg["out_dir"])
    console.print(confirm_table)

    go = questionary.confirm(
        "Run preprocessing with these settings?",
        default=True,
        style=Q_STYLE,
    ).ask()
    if not go:
        console.print("[yellow]Aborted.[/yellow]")
        sys.exit(0)

    console.print()

    # ── Run pipeline ──────────────────────────────────────────
    df_out, original_shape = run_pipeline(df.copy(), cfg)

    # ── Save ──────────────────────────────────────────────────
    with console.status("[cyan]Saving outputs…[/cyan]"):
        saved = save_outputs(df_out, file_path, cfg)

    # ── Summary ───────────────────────────────────────────────
    print_summary(df_out, original_shape, cfg, saved)

    # ── LLM assistant (processed data) ───────────────────────
    want_llm_post = questionary.confirm(
        "Chat with the LLM assistant about the processed data?",
        default=False,
        style=Q_STYLE,
    ).ask()
    if want_llm_post:
        ollama_chat_loop(df_out, stage_label="Processed Data")


if __name__ == "__main__":
    main()
