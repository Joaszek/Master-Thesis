#!/usr/bin/env bash
# setup_data.sh — bootstrap the environment and build the Elliptic2 processed caches.
#
# Installs PyTorch + PyG + torch_scatter (which need custom wheel indexes and cannot come
# from a plain `pip install -r requirements.txt`), then fetches the dataset and builds the
# k-hop subgraph caches.
#
# The dataset ships five CSVs totalling ~88 GB, dominated by background_edges.csv (~83 GB).
# k-hop expansion streams that file with polars scan_csv, so RAM stays modest, but the
# download and the disk footprint do not — budget ~200 GB.
#
# On cloud boxes (Prime Intellect et al.) everything lands under /ephemeral by default,
# and data/processed_k_hop_* is symlinked back into the repo so config.yaml needs no edits.
#
# Usage:
#   bash setup_data.sh                          # deps + download + build k_hop 0 and 1
#   bash setup_data.sh --deps-only              # just install the python environment
#   bash setup_data.sh --skip-deps              # assume the environment is ready
#   bash setup_data.sh --base-dir /mnt/scratch  # different scratch disk
#   bash setup_data.sh --from /path/to/csvs     # reuse CSVs you already have
#   bash setup_data.sh --k-hop 1                # build a single k_hop value
#   bash setup_data.sh --skip-build             # only stage the raw CSVs
#
# Elliptic2 is a public dataset — no Kaggle account or API key is needed. Credentials in
# ~/.kaggle/kaggle.json or KAGGLE_USERNAME + KAGGLE_KEY are used if present.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

KAGGLE_DATASET="ellipticco/elliptic2-data-set"
BASE_DIR=""
RAW_DIR=""
FROM_DIR=""
SKIP_BUILD=0
SKIP_DEPS=0
DEPS_ONLY=0
K_HOPS=(0 1)
PYTHON="${PYTHON:-}"

# Wheel coordinates — must match requirements.txt. Override via env if the cloud image
# ships a different CUDA build.
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
CUDA_TAG="${CUDA_TAG:-cu128}"
SCATTER_VERSION="${SCATTER_VERSION:-2.1.2}"

REQUIRED_FILES=(
    nodes.csv
    edges.csv
    connected_components.csv
    background_nodes.csv
    background_edges.csv
)

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
info()  { echo -e "${CYAN}  $*${RESET}"; }
ok()    { echo -e "${GREEN}  ✔ $*${RESET}"; }
warn()  { echo -e "${YELLOW}  ! $*${RESET}"; }
fail()  { echo -e "${RED}  ✘ $*${RESET}" >&2; exit 1; }
head_() { echo; echo -e "${BOLD}━━ $* ━━${RESET}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-dir)   BASE_DIR="$2"; shift 2 ;;
        --raw-dir)    RAW_DIR="$2"; shift 2 ;;
        --from)       FROM_DIR="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        --skip-deps)  SKIP_DEPS=1; shift ;;
        --deps-only)  DEPS_ONLY=1; shift ;;
        --k-hop)      K_HOPS=("$2"); shift 2 ;;
        -h|--help)    sed -n '2,25p' "$0"; exit 0 ;;
        *)            fail "Unknown argument: $1" ;;
    esac
done

[[ $SKIP_DEPS -eq 1 && $DEPS_ONLY -eq 1 ]] && fail "--deps-only and --skip-deps are mutually exclusive."

# Many images ship python3 without a bare `python`; prefer an active virtualenv if there is one.
if [[ -z "$PYTHON" ]]; then
    for cand in "${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin/python}" python3 python; do
        [[ -n "$cand" ]] && command -v "$cand" >/dev/null 2>&1 && { PYTHON="$cand"; break; }
    done
fi
[[ -n "$PYTHON" ]] || fail "No python interpreter found (tried python3, python). Set PYTHON=/path/to/python."

# ─── 1. Python environment ────────────────────────────────────────────────────
if [[ $SKIP_DEPS -eq 0 ]]; then
    head_ "1/4  Python environment"

    info "python $("$PYTHON" -c 'import sys; print(sys.version.split()[0])') at $("$PYTHON" -c 'import sys; print(sys.executable)')"
    "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)' \
        || fail "Python 3.10+ required."

    "$PYTHON" -m pip --version >/dev/null 2>&1 || fail "pip not available for ${PYTHON}"

    # Debian/Ubuntu images mark the system interpreter as externally managed (PEP 668),
    # which blocks pip outright. Inside a throwaway cloud box that guard is just noise.
    PIP_FLAGS=()
    if [[ -z "${VIRTUAL_ENV:-}" ]] && "$PYTHON" -c 'import os, sys, sysconfig; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_path("stdlib"), "EXTERNALLY-MANAGED")) else 1)'; then
        warn "Externally managed environment (PEP 668) — passing --break-system-packages"
        PIP_FLAGS+=(--break-system-packages)
    fi
    pip_install() { "$PYTHON" -m pip install ${PIP_FLAGS[@]+"${PIP_FLAGS[@]}"} "$@"; }

    pip_install --quiet --upgrade pip

    # Cloud GPU images often preinstall torch. Reusing whatever is there beats forcing a
    # reinstall that can break the driver/CUDA pairing — but torch_scatter must be built
    # against that exact torch, so read the version back rather than assuming the pin.
    if "$PYTHON" -c 'import torch' 2>/dev/null; then
        installed_torch="$("$PYTHON" -c 'import torch; print(torch.__version__)')"
        ok "torch already installed: ${installed_torch}"
        if [[ "$installed_torch" != "${TORCH_VERSION}+${CUDA_TAG}" ]]; then
            warn "differs from the pinned ${TORCH_VERSION}+${CUDA_TAG} — matching torch_scatter to what is installed"
        fi
        TORCH_VERSION="${installed_torch%%+*}"
        case "$installed_torch" in
            *+*) CUDA_TAG="${installed_torch##*+}" ;;
            *)   CUDA_TAG="cpu" ;;
        esac
    else
        info "Installing torch ${TORCH_VERSION} (${CUDA_TAG})"
        pip_install "torch==${TORCH_VERSION}" \
            --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
            || fail "torch installation failed."
        ok "torch ${TORCH_VERSION}+${CUDA_TAG}"
    fi

    # torch_scatter ships local-version wheels (e.g. 2.1.2+pt210cu128) that exist only on
    # the PyG index, keyed by the torch build — plain PyPI resolution cannot find them.
    if "$PYTHON" -c 'import torch_scatter' 2>/dev/null; then
        ok "torch_scatter already installed: $("$PYTHON" -c 'import torch_scatter; print(torch_scatter.__version__)')"
    else
        pyg_index="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"
        info "Installing torch_scatter ${SCATTER_VERSION} from ${pyg_index}"
        pip_install "torch_scatter==${SCATTER_VERSION}" -f "$pyg_index" \
            || fail "torch_scatter installation failed. Check that ${pyg_index} has a wheel for your torch/python combination."
        ok "torch_scatter ${SCATTER_VERSION}"
    fi

    info "Installing remaining requirements"
    pip_install -r requirements.txt || fail "requirements.txt installation failed."

    head_ "Environment check"
    "$PYTHON" - <<'PY' || fail "Environment verification failed."
import importlib, sys
mods = ["torch", "torch_geometric", "torch_scatter", "polars", "sklearn", "kagglehub", "yaml"]
missing = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"    {m:<16} {getattr(mod, '__version__', 'n/a')}")
    except Exception as e:
        missing.append(f"{m} ({e})")
if missing:
    sys.exit("    MISSING: " + ", ".join(missing))

import torch
print(f"    {'cuda available':<16} {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    {'gpu':<16} {torch.cuda.get_device_name(0)}")
else:
    print("    WARNING: no CUDA device — training and attacks require a GPU.")
PY
    ok "Environment ready"

    if [[ $DEPS_ONLY -eq 1 ]]; then
        echo; info "--deps-only: stopping before data setup."
        exit 0
    fi
else
    head_ "1/4  Python environment — skipped (--skip-deps)"
fi

# ─── Where everything lives ───────────────────────────────────────────────────
# /ephemeral is the scratch disk on Prime Intellect; fall back to the repo when absent.
if [[ -z "$BASE_DIR" ]]; then
    if [[ -d /ephemeral && -w /ephemeral ]]; then
        BASE_DIR="/ephemeral/elliptic2"
    else
        BASE_DIR="${REPO_ROOT}/data"
    fi
fi
mkdir -p "$BASE_DIR"
BASE_DIR="$(cd "$BASE_DIR" && pwd)"
[[ -n "$RAW_DIR" ]] || RAW_DIR="${BASE_DIR}/raw"
export KAGGLEHUB_CACHE="${KAGGLEHUB_CACHE:-${BASE_DIR}/kagglehub}"

# ─── 2. Stage the raw CSVs ────────────────────────────────────────────────────
head_ "2/4  Raw data → ${RAW_DIR}"
printf "  %-18s %s\n" "base dir"  "$BASE_DIR"
printf "  %-18s %s\n" "kagglehub" "$KAGGLEHUB_CACHE"
if [[ "$BASE_DIR" == /ephemeral* ]]; then
    warn "/ephemeral is wiped when the instance is torn down — copy data/processed_k_hop_* off before shutdown."
fi
mkdir -p "$RAW_DIR"

link_in() {
    # link_in <source-file> <basename> — hard-link when possible, else symlink, else copy.
    local src="$1" name="$2" dst="${RAW_DIR}/$2"
    [[ -s "$dst" ]] && { ok "${name} (already staged)"; return; }
    ln "$src" "$dst" 2>/dev/null \
        || ln -s "$(realpath "$src")" "$dst" 2>/dev/null \
        || cp "$src" "$dst" \
        || fail "Could not stage ${name}"
    ok "$name"
}

missing=()
for f in "${REQUIRED_FILES[@]}"; do
    [[ -s "${RAW_DIR}/${f}" ]] || missing+=("$f")
done

if [[ ${#missing[@]} -eq 0 ]]; then
    ok "All ${#REQUIRED_FILES[@]} CSVs already present — nothing to fetch"

elif [[ -n "$FROM_DIR" ]]; then
    [[ -d "$FROM_DIR" ]] || fail "--from directory not found: $FROM_DIR"
    info "Staging ${#missing[@]} missing file(s) from ${FROM_DIR}"
    for f in "${missing[@]}"; do
        src="${FROM_DIR}/${f}"
        [[ -s "$src" ]] || fail "Missing in ${FROM_DIR}: ${f}"
        link_in "$src" "$f"
    done

else
    "$PYTHON" -c 'import kagglehub' 2>/dev/null \
        || fail "kagglehub not installed. Run without --skip-deps, or: pip install kagglehub"

    # Elliptic2 is public (CC BY-NC-ND 4.0) and kagglehub fetches it anonymously.
    # Credentials are picked up if present but are not required.
    if [[ -f "${HOME:-/nonexistent}/.kaggle/kaggle.json" || -n "${KAGGLE_USERNAME:-}" ]]; then
        info "Kaggle credentials detected — will be used"
    else
        info "No Kaggle credentials — downloading anonymously (the dataset is public)"
    fi

    avail_gb=$(df -BG --output=avail "$BASE_DIR" | tail -1 | tr -dc '0-9')
    info "Free space on ${BASE_DIR}: ${avail_gb} GB"
    (( avail_gb < 200 )) && warn "Elliptic2 needs ~88 GB extracted plus transient archive space; 200+ GB recommended."

    info "Downloading ${KAGGLE_DATASET} via kagglehub — this takes a while (~88 GB)"
    DATASET_PATH="$(
        "$PYTHON" - <<'PY'
import kagglehub
path = kagglehub.dataset_download("ellipticco/elliptic2-data-set")
print(path)
PY
    )" || fail "kagglehub download failed."
    [[ -d "$DATASET_PATH" ]] || fail "kagglehub returned a path that is not a directory: ${DATASET_PATH}"
    ok "Downloaded to ${DATASET_PATH}"

    # kagglehub may nest the CSVs inside the extracted tree — locate each one.
    info "Staging CSVs into ${RAW_DIR}"
    for f in "${missing[@]}"; do
        src="$(find "$DATASET_PATH" -type f -name "$f" -print -quit)"
        [[ -n "$src" ]] || fail "Not found in downloaded dataset: ${f}"
        link_in "$src" "$f"
    done
fi

# ─── 3. Verify ────────────────────────────────────────────────────────────────
head_ "3/4  Verifying raw files"
for f in "${REQUIRED_FILES[@]}"; do
    [[ -s "${RAW_DIR}/${f}" ]] || fail "Missing or empty: ${RAW_DIR}/${f}"
    printf "  %-28s %s\n" "$f" "$(du -Lh "${RAW_DIR}/${f}" | cut -f1)"
done
ok "All ${#REQUIRED_FILES[@]} required CSVs present"

# ─── 4. Build processed subgraphs ─────────────────────────────────────────────
if [[ $SKIP_BUILD -eq 1 ]]; then
    head_ "4/4  Build skipped (--skip-build)"
    echo
    info "Build with:  ${PYTHON} -m src.preprocess.preprocess --k-hop 1 --raw-dir ${RAW_DIR} --processed-dir ${BASE_DIR}/processed_k_hop_1"
    exit 0
fi

head_ "4/4  Building processed subgraphs for k_hop: ${K_HOPS[*]}"
warn "k-hop expansion streams background_edges.csv (~83 GB) once per hop — expect this to be slow."

for k in "${K_HOPS[@]}"; do
    out="${BASE_DIR}/processed_k_hop_${k}"
    echo
    info "── k_hop=${k} → ${out}"
    "$PYTHON" -m src.preprocess.preprocess \
        --k-hop "$k" --raw-dir "$RAW_DIR" --processed-dir "$out" \
        || fail "Preprocessing failed for k_hop=${k}"

    # config.yaml points at data/processed_k_hop_*; symlink so it works unchanged
    # when the real data lives on a scratch disk.
    repo_link="${REPO_ROOT}/data/processed_k_hop_${k}"
    if [[ "$out" != "$repo_link" ]]; then
        mkdir -p "${REPO_ROOT}/data"
        [[ -L "$repo_link" ]] && rm "$repo_link"
        [[ -e "$repo_link" ]] && fail "Refusing to overwrite existing directory: ${repo_link} (move it first)"
        ln -s "$out" "$repo_link"
        ok "linked ${repo_link} → ${out}"
    fi

    # For k_hop>0 the expansion flags must survive into the parquet, otherwise
    # expansion_node_weight silently degrades to a no-op.
    if (( k > 0 )); then
        "$PYTHON" - "$out" <<'PY'
import sys, polars as pl
path = f"{sys.argv[1]}/nodes.parquet"
df = pl.read_parquet(path)
if "is_original" not in df.columns:
    sys.exit(f"    FAIL: {path} has no is_original column (columns: {df.columns}).\n"
             f"          It was written by an older preprocess.py — delete it and rebuild.")
counts = df["is_original"].value_counts()
d = dict(zip(counts["is_original"].to_list(), counts["count"].to_list()))
n_orig, n_exp = d.get(True, 0), d.get(False, 0)
print(f"    is_original: {n_orig:,} original | {n_exp:,} expansion")
if n_exp == 0:
    sys.exit("    FAIL: no expansion nodes flagged — expansion_node_weight would be a no-op")
PY
        ok "expansion flags verified"
    fi
    ok "k_hop=${k} done"
done

echo
head_ "Done"
cat <<EOF

  Processed data:  ${BASE_DIR}/processed_k_hop_*
  Repo symlinks:   data/processed_k_hop_*  (config.yaml works unchanged)

  Next:
    ${PYTHON} -m src.train.train          # train all four architectures, 3 seeds
    bash run_all_attacks.sh               # full adversarial evaluation

EOF
