#!/usr/bin/env bash
# =============================================================================
#  DataPrepX — Installer Script
#  Enhancing quality through pre-processing techniques in Data Analytics
# =============================================================================
#
#  Usage:
#    chmod +x install.sh
#    ./install.sh              # standard install
#    ./install.sh --venv       # install inside a virtual environment (recommended)
#    ./install.sh --help       # show this help
#
# =============================================================================

set -euo pipefail

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

ok()   { echo -e "${GREEN}  ✔  ${RESET}$*"; }
info() { echo -e "${CYAN}  ●  ${RESET}$*"; }
warn() { echo -e "${YELLOW}  ⚠  ${RESET}$*"; }
fail() { echo -e "${RED}  ✘  ${RESET}$*" >&2; exit 1; }
step() { echo -e "\n${BOLD}${CYAN}▶  $*${RESET}"; }

# ── Banner ─────────────────────────────────────────────────────────────────────
print_banner() {
    echo -e "${CYAN}"
    echo "  ╔══════════════════════════════════════════════════════════════╗"
    echo "  ║              DataPrepX  —  Installer                        ║"
    echo "  ║    Smart Data Preprocessing · CLI + Gradio + Ollama         ║"
    echo "  ╚══════════════════════════════════════════════════════════════╝"
    echo -e "${RESET}"
}

# ── Usage / Help ───────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
${BOLD}Usage:${RESET}
  ./install.sh [OPTIONS]

${BOLD}Options:${RESET}
  --venv         Create and use a Python virtual environment (.venv/)
  --venv-dir DIR Use a custom directory for the virtual environment
  --skip-ollama  Skip the optional Ollama (LLM) installation check
  --dev          Also install optional development/testing dependencies
  --help         Show this help message and exit

${BOLD}Examples:${RESET}
  ./install.sh
  ./install.sh --venv
  ./install.sh --venv --venv-dir ~/envs/dataprepx
  ./install.sh --skip-ollama
EOF
    exit 0
}

# ── Argument parsing ───────────────────────────────────────────────────────────
USE_VENV=false
VENV_DIR=".venv"
SKIP_OLLAMA=false
INSTALL_DEV=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)         USE_VENV=true; shift ;;
        --venv-dir)     USE_VENV=true; VENV_DIR="$2"; shift 2 ;;
        --skip-ollama)  SKIP_OLLAMA=true; shift ;;
        --dev)          INSTALL_DEV=true; shift ;;
        --help|-h)      usage ;;
        *) fail "Unknown option: $1  (run ./install.sh --help for usage)" ;;
    esac
done

# ── Script directory (repo root) ───────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f "requirements.txt" ]] || [[ ! -f "cli.py" ]]; then
    step "Project files not found. Cloning repository..."
    REPO_URL="https://github.com/shreyas23dev/DataPrepX-A-tool-for-Enhancing-quality-through-pre-processing-technique-in-Data-Analytics.git"
    CLONE_DIR="DataPrepX"
    
    if ! command -v git &>/dev/null; then
        fail "git is not installed. Please install git or download the repository manually."
    fi
    
    git clone "$REPO_URL" "$CLONE_DIR" || fail "Failed to clone repository."
    cd "$CLONE_DIR"
    SCRIPT_DIR="$(pwd)"
    CLONED_INTO="$CLONE_DIR"
    ok "Repository cloned into $SCRIPT_DIR"
fi

# =============================================================================
print_banner

# ── 1. Python check ────────────────────────────────────────────────────────────
step "Checking Python installation"

PYTHON_BIN=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        PY_VER=$("$candidate" -c 'import sys; print(sys.version_info[:2])')
        if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)' 2>/dev/null; then
            PYTHON_BIN="$candidate"
            ok "Found $("$candidate" --version) at $(command -v "$candidate")"
            break
        else
            warn "Found $("$candidate" --version) but DataPrepX requires Python ≥ 3.9"
        fi
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    fail "Python 3.9+ is required but was not found.\n  Install it from https://www.python.org/downloads/ and re-run this script."
fi

# ── 2. pip check ───────────────────────────────────────────────────────────────
step "Checking pip"

PIP_CMD=""
if "$PYTHON_BIN" -m pip --version &>/dev/null; then
    PIP_CMD="$PYTHON_BIN -m pip"
elif command -v pip3 &>/dev/null; then
    PIP_CMD="pip3"
elif command -v pip &>/dev/null; then
    PIP_CMD="pip"
else
    warn "pip is not available — attempting bootstrap via ensurepip"
    if ! "$PYTHON_BIN" -m ensurepip --upgrade &>/dev/null; then
        if $USE_VENV; then
            warn "Global pip not found, but will attempt to use virtual environment pip."
        else
            fail "Could not bootstrap pip. Please install pip manually (e.g., sudo apt install python3-pip) and retry."
        fi
    else
        PIP_CMD="$PYTHON_BIN -m pip"
    fi
fi

if [[ -n "$PIP_CMD" ]]; then
    ok "pip is available  →  $($PIP_CMD --version)"
fi

# ── 3. Virtual environment ─────────────────────────────────────────────────────
if $USE_VENV; then
    step "Setting up virtual environment at: ${BOLD}$VENV_DIR${RESET}"

    if [[ ! -d "$VENV_DIR" ]]; then
        "$PYTHON_BIN" -m venv "$VENV_DIR"
        ok "Virtual environment created"
    else
        info "Virtual environment already exists — reusing it"
    fi

    # Activate
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    PYTHON_BIN="python"
    PIP_CMD="$PYTHON_BIN -m pip"
    ok "Activated: $(python --version)  ($VENV_DIR)"
fi

# ── 4. Upgrade pip + setuptools inside the env ────────────────────────────────
step "Upgrading pip & setuptools"
$PIP_CMD install --upgrade pip setuptools wheel -q
ok "pip, setuptools, wheel are up-to-date"

# ── 5. Install project dependencies ───────────────────────────────────────────
step "Installing DataPrepX dependencies (requirements.txt)"

if [[ ! -f "$SCRIPT_DIR/requirements.txt" ]]; then
    fail "requirements.txt not found in $SCRIPT_DIR"
fi

$PIP_CMD install -r "$SCRIPT_DIR/requirements.txt"
ok "All dependencies installed successfully"

# ── 6. Optional dev dependencies ──────────────────────────────────────────────
if $INSTALL_DEV; then
    step "Installing optional development dependencies"
    $PIP_CMD install pytest pytest-cov black isort mypy -q
    ok "Development tools installed (pytest, black, isort, mypy)"
fi

# ── 7. Verify critical imports ─────────────────────────────────────────────────
step "Verifying critical imports"

IMPORTS=(pandas numpy sklearn rich questionary openpyxl)
FAILED_IMPORTS=()

for pkg in "${IMPORTS[@]}"; do
    if "$PYTHON_BIN" -c "import ${pkg}" 2>/dev/null; then
        ok "${pkg}"
    else
        warn "Import failed: ${pkg}"
        FAILED_IMPORTS+=("$pkg")
    fi
done

if [[ ${#FAILED_IMPORTS[@]} -gt 0 ]]; then
    fail "The following packages could not be imported: ${FAILED_IMPORTS[*]}\n  Try running: $PYTHON_BIN -m pip install ${FAILED_IMPORTS[*]}"
fi

# ── 8. Ollama (optional LLM backend) ──────────────────────────────────────────
if ! $SKIP_OLLAMA; then
    step "Checking Ollama (optional — required for LLM assistant feature)"

    if command -v ollama &>/dev/null; then
        OLLAMA_VER=$(ollama --version 2>/dev/null || echo "unknown")
        ok "Ollama is installed  →  $OLLAMA_VER"

        if curl -s --connect-timeout 3 http://localhost:11434/api/tags &>/dev/null; then
            ok "Ollama server is running at http://localhost:11434"
        else
            warn "Ollama is installed but the server does not appear to be running."
            echo -e "      ${DIM}Start it with:  ollama serve${RESET}"
        fi
    else
        warn "Ollama is NOT installed."
        echo -e "      ${DIM}The CLI's LLM assistant feature requires Ollama."
        echo -e "      Install from: https://ollama.com/download"
        echo -e "      Then pull a model, e.g.:  ollama pull llama3.2${RESET}"
    fi
fi

# ── 9. Make cli.py executable ──────────────────────────────────────────────────
step "Making cli.py executable"

if [[ -f "cli.py" ]]; then
    chmod +x cli.py
    ok "cli.py is now executable"
else
    warn "cli.py not found — skipping chmod"
fi

# ── 10. (Optional) Create a convenience launcher script ───────────────────────
step "Creating launcher script: dataprepx"

LAUNCHER_PATH="$SCRIPT_DIR/dataprepx"

if $USE_VENV; then
    # Launcher that auto-activates the venv
    cat > "$LAUNCHER_PATH" <<LAUNCHER
#!/usr/bin/env bash
# DataPrepX launcher — auto-activates the virtual environment
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
source "\$SCRIPT_DIR/$VENV_DIR/bin/activate"
exec python "\$SCRIPT_DIR/cli.py" "\$@"
LAUNCHER
else
    cat > "$LAUNCHER_PATH" <<LAUNCHER
#!/usr/bin/env bash
# DataPrepX launcher
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BIN}" "\$SCRIPT_DIR/cli.py" "\$@"
LAUNCHER
fi

chmod +x "$LAUNCHER_PATH"
ok "Launcher created at: $LAUNCHER_PATH"

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}  ════════════════════════════════════════════════════════════${RESET}"
echo -e "${GREEN}${BOLD}  ✔  DataPrepX installation complete!${RESET}"
echo -e "${CYAN}  ════════════════════════════════════════════════════════════${RESET}"
echo ""
echo -e "  ${BOLD}Run DataPrepX:${RESET}"

if [[ -n "${CLONED_INTO:-}" ]]; then
    echo -e "    ${YELLOW}cd ${CLONED_INTO}${RESET}"
fi

if $USE_VENV; then
    echo -e "    ${DIM}Option 1 (launcher, no activation needed):${RESET}"
    echo -e "      ${CYAN}./dataprepx${RESET}"
    echo -e "    ${DIM}Option 2 (manual):${RESET}"
    echo -e "      ${CYAN}source ${VENV_DIR}/bin/activate${RESET}"
    echo -e "      ${CYAN}python cli.py${RESET}"
else
    echo -e "      ${CYAN}./dataprepx${RESET}"
    echo -e "    ${DIM}or${RESET}"
    echo -e "      ${CYAN}${PYTHON_BIN} cli.py${RESET}"
fi

echo ""
echo -e "  ${BOLD}Pass a file directly:${RESET}"
echo -e "      ${CYAN}./dataprepx --file path/to/data.csv${RESET}"
echo ""
echo -e "  ${DIM}Documentation: README.md${RESET}"
echo ""
