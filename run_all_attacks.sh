#!/usr/bin/env bash
# run_all_attacks.sh — Runs all adversarial attacks sequentially.
# Usage:
#   bash run_all_attacks.sh            # all attacks
#   bash run_all_attacks.sh pgd zoo    # only selected attacks
#
# Attacks: pgd | transfer | saliency | saliency_ig | topology | injection | zoo

set -euo pipefail

# ─── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

# ─── Helpers ──────────────────────────────────────────────────────────────────
PASS=0; FAIL=0; SKIP=0
declare -A RESULTS   # attack_name -> PASS | FAIL | SKIP

run_attack() {
    local name="$1"; shift
    local cmd=("$@")

    echo ""
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo -e "${CYAN}${BOLD}  ▶  ${name}${RESET}"
    echo -e "${CYAN}  $(date '+%H:%M:%S')  Command: ${cmd[*]}${RESET}"
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"

    local t0=$SECONDS
    if "${cmd[@]}"; then
        local elapsed=$(( SECONDS - t0 ))
        echo -e "${GREEN}  ✔  ${name} — done in ${elapsed}s${RESET}"
        RESULTS[$name]="PASS"
        (( PASS++ )) || true
    else
        local elapsed=$(( SECONDS - t0 ))
        echo -e "${RED}  ✘  ${name} — FAILED after ${elapsed}s${RESET}"
        RESULTS[$name]="FAIL"
        (( FAIL++ )) || true
        if [[ "${STOP_ON_FAIL:-0}" == "1" ]]; then
            echo -e "${RED}Stopping due to STOP_ON_FAIL=1${RESET}"
            exit 1
        fi
    fi
}

should_run() {
    # If no filter args given, run everything.
    [[ ${#FILTER[@]} -eq 0 ]] && return 0
    local name="$1"
    for f in "${FILTER[@]}"; do
        [[ "$f" == "$name" ]] && return 0
    done
    return 1
}

# ─── Parse optional filter list ───────────────────────────────────────────────
FILTER=("${@}")

# ─── Attack definitions ───────────────────────────────────────────────────────

if should_run "pgd"; then
    run_attack "pgd" \
        python scripts/pgd/run_pgd_attack.py
else
    RESULTS["pgd"]="SKIP"; (( SKIP++ )) || true
fi

if should_run "transfer"; then
    run_attack "transfer" \
        python scripts/transfer/run_transfer_attack.py
else
    RESULTS["transfer"]="SKIP"; (( SKIP++ )) || true
fi

# Saliency — default (gradient)
if should_run "saliency"; then
    run_attack "saliency" \
        python scripts/saliency/run_saliency_attack.py
else
    RESULTS["saliency"]="SKIP"; (( SKIP++ )) || true
fi

# Saliency — Integrated Gradients variant
if should_run "saliency_ig"; then
    run_attack "saliency_ig" \
        python scripts/saliency/run_saliency_attack.py \
            --importance_method ig \
            --output data/results/saliency_ig_attack_results.json \
            --plots_dir data/results/plots/saliency_ig
else
    RESULTS["saliency_ig"]="SKIP"; (( SKIP++ )) || true
fi

if should_run "topology"; then
    run_attack "topology" \
        python scripts/topology/run_topology_attack.py
else
    RESULTS["topology"]="SKIP"; (( SKIP++ )) || true
fi

if should_run "injection"; then
    run_attack "injection" \
        python scripts/injection/run_injection_attack.py
else
    RESULTS["injection"]="SKIP"; (( SKIP++ )) || true
fi

if should_run "zoo"; then
    run_attack "zoo" \
        python scripts/zoo/run_zoo_attack.py
else
    RESULTS["zoo"]="SKIP"; (( SKIP++ )) || true
fi

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${BOLD}  SUMMARY${RESET}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
for attack in pgd transfer saliency saliency_ig topology injection zoo; do
    status="${RESULTS[$attack]:-SKIP}"
    case "$status" in
        PASS) echo -e "  ${GREEN}✔ ${attack}${RESET}" ;;
        FAIL) echo -e "  ${RED}✘ ${attack}${RESET}" ;;
        SKIP) echo -e "  ${YELLOW}– ${attack} (skipped)${RESET}" ;;
    esac
done
echo ""
echo -e "  Passed: ${GREEN}${PASS}${RESET}  Failed: ${RED}${FAIL}${RESET}  Skipped: ${YELLOW}${SKIP}${RESET}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"

[[ $FAIL -eq 0 ]]
