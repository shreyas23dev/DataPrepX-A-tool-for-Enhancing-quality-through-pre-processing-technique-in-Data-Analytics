# DataPrepX — Intelligent Terminal-based Data Preprocessing

DataPrepX is an open-source, interactive data-preprocessing toolkit designed to streamline the transition from raw data to model-ready features. Built for power users and automation pipelines, it provides a guided CLI experience that ensures reproducibility through JSON configuration tracking.

---

## 🚀 New in v2.0: The Intelligence Update

The latest release introduces LLM-augmented data engineering and automated predictive insights:

- 🤖 **Conversational AI Assistant** — Integrated with local Ollama servers to allow natural-language querying of your dataset.
- 🌲 **Random Forest Integration** — Built-in support for baseline predictive scoring and feature importance rankings.
- 🛠️ **AI Tool-Calling** — The assistant can perform cell-level edits and query data via a secure, restricted `eval()` sandbox.
- ⚡ **Streaming & Context** — Support for token-by-token streaming and an extended 30,000-token context window via a custom `Modelfile`.

---

## ✨ Key Features

- **Interactive Terminal CLI** — A fully guided experience using `Rich` and `Questionary` with autocomplete and live dataset previews.

- **6-Step Preprocessing Pipeline:**

  1. **Target Isolation** — Protects the label column from transformations.
  2. **Missing Value Handling** — Seven strategies including mean, median, and backfill.
  3. **Categorical Encoding** — Label, One-Hot, and Ordinal encoding.
  4. **Outlier Removal** — Univariate IQR filters and multivariate Elliptic Envelope detection.
  5. **Feature Selection** — `VarianceThreshold` to drop near-constant columns.
  6. **Feature Scaling** — Standard, MinMax, and Robust scaling.

- **Reproducibility** — Every session generates a `job_config.json` recording every decision made.
- **Multi-Format Export** — Generates CSV, JSON, and XLSX outputs in a single pass.

---

## 📦 Installation & Usage

### 1. Automated Setup

The provided installer creates a virtual environment and handles all dependencies.

```bash
bash installer.sh
```

### 2. Initialize the AI Assistant

Ensure you have [Ollama](https://ollama.com/) installed, then build the custom model.

```bash
ollama create dataprepx-assistant -f Modelfile
```

### 3. Launch the CLI

```bash
source .venv/bin/activate
python cli.py
```

---

## 🛠️ Technology Stack

| Layer | Libraries |
|---|---|
| **Data** | `pandas`, `numpy`, `scikit-learn` |
| **Interface** | `rich`, `questionary` |
| **AI / LLM** | `Ollama` (llama3.2), `requests` |
| **Analysis** | `ydata-profiling` |

---

## 👥 Developed By

**Shreyas A**  and **Trinath Bhattacharya @https://github.com/CodingLangur **

