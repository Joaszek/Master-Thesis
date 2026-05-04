import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats as sp_stats
from scipy.stats import chi2

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
PLOTS   = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

MODEL_KEYS    = ["gatv2", "sage", "sage_edge", "gin"]
MODEL_LABELS  = {"gatv2": "GATv2", "sage": "SAGE",
                 "sage_edge": "SAGE+Edge", "gin": "GIN"}
MODEL_COLORS  = {"gatv2": "#e41a1c", "sage": "#377eb8",
                 "sage_edge": "#4daf4a", "gin": "#984ea3"}
ATTACK_LABELS = {
    "pgd":           "PGD (white-box)",
    "zoo":           "ZOO (zoo)",
    "transfer":      "Transfer (gray-box)",
    "saliency_grad": "Saliency-Grad (white-box)",
    "saliency_ig":   "Saliency-IG (white-box)",
    "injection":     "Node Injection",
    "rewiring":      "Topology Rewiring",
}
ATTACK_COLORS = {
    "pgd":           "#1b9e77",
    "zoo":           "#d95f02",
    "transfer":      "#7570b3",
    "saliency_grad": "#e7298a",
    "saliency_ig":   "#f781bf",
    "injection":     "#66a61e",
    "rewiring":      "#e6ab02",
}
ATTACK_SHORT = {
    "pgd":           "PGD",
    "zoo":           "ZOO",
    "transfer":      "Transfer",
    "saliency_grad": "Sal-Grad",
    "saliency_ig":   "Sal-IG",
    "injection":     "Node Inj.",
    "rewiring":      "Topology",
}

plt.rcParams.update({
    "figure.dpi":       150,
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
})


def load_json(name):
    p = RESULTS / name
    with open(p) as f:
        return json.load(f)


pgd_raw      = load_json("pgd_attack_results.json")
zoo_raw      = load_json("zoo_attack_results.json")
transfer_raw = load_json("transfer_attack_results.json")
sal_raw      = load_json("saliency_attack_results.json")
sal_ig_raw   = load_json("saliency_ig_attack_results.json")
inj_raw      = load_json("node_injection_results.json")
topo_raw     = load_json("topology_rewiring_results.json")
_baseline_path = RESULTS / "multi_seed_summary.json"
if _baseline_path.exists():
    baseline_raw = load_json("multi_seed_summary.json")
else:
    baseline_raw = {}
    for m in MODEL_KEYS:
        eps_vals = pgd_raw["models"][m]["epsilon_results"].values()
        f1_vals  = [v["clean_f1_macro"] for v in eps_vals]
        mean_f1  = float(np.mean(f1_vals))
        baseline_raw[MODEL_LABELS[m]] = {
            "f1_macro": {"mean": mean_f1, "std": float(np.std(f1_vals)),
                         "min": float(np.min(f1_vals)), "max": float(np.max(f1_vals))}
        }
    print("  [WARNING] multi_seed_summary.json not found — "
          "baseline F1 estimated from PGD clean_f1_macro values.")

print("All result files loaded.")


CANON_EPS   = "0.1"
CANON_INJ   = "random_k3_c1"
CANON_TOPO  = "chain_injection_k5"
CANON_SAL_K = "5"

def get_pgd(model, eps=CANON_EPS):
    r = pgd_raw["models"][model]["epsilon_results"][eps]
    return dict(asr=r["asr"],
                f1_drop=abs(r["clean_f1_macro"] - r["post_f1_macro"]),
                budget=float(eps),
                n_evaded=r["n_evaded"],
                n_attacked=r["n_attacked"],
                conf_pre=r["mean_confidence_pre"],
                conf_post=r["mean_confidence_post"])

def get_zoo(model, eps=CANON_EPS):
    r = zoo_raw["models"][model]["epsilon_results"][eps]
    return dict(asr=r["asr"],
                f1_drop=abs(r["clean_f1_macro"] - r["post_f1_macro"]),
                budget=float(eps),
                n_evaded=r["n_evaded"],
                n_attacked=r["n_attacked"],
                conf_pre=r["mean_confidence_pre"],
                conf_post=r["mean_confidence_post"],
                mean_queries=r["mean_queries_used"])

def get_transfer(model, eps=CANON_EPS):
    r = transfer_raw["gcn_to_targets"][eps][model]
    return dict(asr=r["transfer_asr"],
                f1_drop=abs(r["clean_f1_macro"] - r["post_f1_macro"]),
                budget=float(eps),
                n_evaded=r["n_evaded"],
                n_attacked=r["n_attacked"],
                conf_pre=r["mean_confidence_pre"],
                conf_post=r["mean_confidence_post"])

def get_saliency_grad(model, topk=CANON_SAL_K, eps=CANON_EPS):
    r = sal_raw["models"][model]["topk_results"][topk]["epsilon_results"][eps]
    return dict(asr=r["asr"],
                f1_drop=abs(r["clean_f1_macro"] - r["post_f1_macro"]),
                budget=float(eps),
                n_evaded=r["n_evaded"],
                n_attacked=r["n_attacked"],
                conf_pre=r["mean_confidence_pre"],
                conf_post=r["mean_confidence_post"])

def get_saliency_ig(model, topk=CANON_SAL_K, eps=CANON_EPS):
    r = sal_ig_raw["models"][model]["topk_results"][topk]["epsilon_results"][eps]
    return dict(asr=r["asr"],
                f1_drop=abs(r["clean_f1_macro"] - r["post_f1_macro"]),
                budget=float(eps),
                n_evaded=r["n_evaded"],
                n_attacked=r["n_attacked"],
                conf_pre=r["mean_confidence_pre"],
                conf_post=r["mean_confidence_post"])

def get_injection(model, combo=CANON_INJ):
    r = inj_raw["models"][model]["combinations"][combo]
    budget = r["k_nodes"] * r["connections_per_node"]
    return dict(asr=r["asr"],
                f1_drop=abs(r["clean_f1_macro"] - r["post_f1_macro"]),
                budget=float(budget),
                n_evaded=r["n_evaded"],
                n_attacked=r["n_attacked"],
                conf_pre=r["mean_confidence_pre"],
                conf_post=r["mean_confidence_post"])

def get_rewiring(model, combo=CANON_TOPO):
    r = topo_raw["models"][model]["combinations"][combo]
    budget = r["mean_delta_nodes"] + r["mean_delta_edges"]
    return dict(asr=r["asr"],
                f1_drop=abs(r["clean_f1_macro"] - r["post_f1_macro"]),
                budget=float(budget),
                n_evaded=r["n_evaded"],
                n_attacked=r["n_attacked"],
                conf_pre=r["mean_confidence_pre"],
                conf_post=r["mean_confidence_post"])


ATTACKS = ["pgd", "zoo", "transfer", "saliency_grad", "saliency_ig", "injection", "rewiring"]
GETTERS = dict(pgd=get_pgd, zoo=get_zoo, transfer=get_transfer,
               saliency_grad=get_saliency_grad, saliency_ig=get_saliency_ig,
               injection=get_injection, rewiring=get_rewiring)

master = {}
for m in MODEL_KEYS:
    master[m] = {}
    for atk in ATTACKS:
        master[m][atk] = GETTERS[atk](m)

print("Canonical metrics extracted.")


def max_asr_pgd(model):
    return max(v["asr"] for v in pgd_raw["models"][model]["epsilon_results"].values())

def max_asr_zoo(model):
    return max(v["asr"] for v in zoo_raw["models"][model]["epsilon_results"].values())

def max_asr_transfer(model):
    return max(v["transfer_asr"]
               for v in transfer_raw["gcn_to_targets"].values()
               for k, v2 in v.items() if k == model
               for v in [v2])

def max_asr_saliency_grad(model):
    best = 0.0
    for tk_data in sal_raw["models"][model]["topk_results"].values():
        for eps_data in tk_data["epsilon_results"].values():
            best = max(best, eps_data["asr"])
    return best

def max_asr_saliency_ig(model):
    best = 0.0
    for tk_data in sal_ig_raw["models"][model]["topk_results"].values():
        for eps_data in tk_data["epsilon_results"].values():
            best = max(best, eps_data["asr"])
    return best

def max_asr_injection(model):
    return max(v["asr"] for v in inj_raw["models"][model]["combinations"].values())

def max_asr_rewiring(model):
    return max(v["asr"] for v in topo_raw["models"][model]["combinations"].values())


MAX_ASR_FNS = dict(pgd=max_asr_pgd, zoo=max_asr_zoo, transfer=max_asr_transfer,
                   saliency_grad=max_asr_saliency_grad, saliency_ig=max_asr_saliency_ig,
                   injection=max_asr_injection, rewiring=max_asr_rewiring)

max_asr = {}
for m in MODEL_KEYS:
    max_asr[m] = {atk: MAX_ASR_FNS[atk](m) for atk in ATTACKS}


N_MODEL_PAIRS    = len(MODEL_KEYS) * (len(MODEL_KEYS) - 1) // 2  # = 6
BONFERRONI_ALPHA = 0.05 / N_MODEL_PAIRS
 

def two_proportion_ztest(n1_success, n1_total, n2_success, n2_total):
    if n1_total == 0 or n2_total == 0:
        return float("nan"), float("nan")
 
    p1 = n1_success / n1_total
    p2 = n2_success / n2_total
    p_pool = (n1_success + n2_success) / (n1_total + n2_total)
 
    if p_pool == 0.0 or p_pool == 1.0:
        return float("nan"), float("nan")
 
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1_total + 1/n2_total))
    if se == 0:
        return float("nan"), float("nan")
 
    correction = 0.5 * (1/n1_total + 1/n2_total)
    z = (abs(p1 - p2) - correction) / se
    z = max(z, 0.0)
 
    p_value = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    return z, p_value

 
def mcnemar_exact(b, c):
    if b + c == 0:
        return float("nan"), float("nan")
    chi = (abs(b - c) - 1.0) ** 2 / (b + c)
    p = 1.0 - chi2.cdf(chi, df=1)
    return chi, p
 
 
def _load_per_graph_for_attack(atk):
    result = {}
 
    for m in MODEL_KEYS:
        try:
            if atk == "pgd":
                eps_data = pgd_raw["models"][m]["epsilon_results"][CANON_EPS]
                pg = eps_data.get("per_graph", {})
            elif atk == "zoo":
                eps_data = zoo_raw["models"][m]["epsilon_results"][CANON_EPS]
                pg = eps_data.get("per_graph", {})
            elif atk == "transfer":
                eps_data = transfer_raw["gcn_to_targets"][CANON_EPS][m]
                pg = eps_data.get("per_graph", {})
            elif atk == "saliency_grad":
                eps_data = (sal_raw["models"][m]["topk_results"][CANON_SAL_K]
                            ["epsilon_results"][CANON_EPS])
                pg = eps_data.get("per_graph", {})
            elif atk == "saliency_ig":
                eps_data = (sal_ig_raw["models"][m]["topk_results"][CANON_SAL_K]
                            ["epsilon_results"][CANON_EPS])
                pg = eps_data.get("per_graph", {})
            elif atk == "injection":
                combo_data = inj_raw["models"][m]["combinations"][CANON_INJ]
                pg = combo_data.get("per_graph", {})
            elif atk == "rewiring":
                combo_data = topo_raw["models"][m]["combinations"][CANON_TOPO]
                pg = combo_data.get("per_graph", {})
            else:
                return None
 
            indices = pg.get("subgraph_indices")
            evaded  = pg.get("evaded")
 
            if indices is None or evaded is None:
                print(f"  [McNemar] Missing per_graph data for {atk}/{m} — skipping McNemar")
                return None
 
            if len(indices) != len(evaded):
                print(f"  [McNemar] Length mismatch in per_graph for {atk}/{m} — skipping")
                return None
 
            result[m] = dict(zip(indices, evaded))
 
        except (KeyError, TypeError) as e:
            print(f"  [McNemar] Could not load per_graph for {atk}/{m}: {e}")
            return None
 
    return result
 
 
def compute_mcnemar_for_pair(per_graph_m1, per_graph_m2):
    shared = set(per_graph_m1.keys()) & set(per_graph_m2.keys())
    n_shared = len(shared)
 
    if n_shared == 0:
        return {
            "intersection_size": 0,
            "b": 0, "c": 0,
            "chi2": None, "p_value": None,
            "significant_uncorrected": False,
            "significant_bonferroni":  False,
            "note": "Empty intersection — no shared TP subgraphs",
        }

    b = sum(1 for i in shared if per_graph_m1[i] == 1 and per_graph_m2[i] == 0)
    c = sum(1 for i in shared if per_graph_m1[i] == 0 and per_graph_m2[i] == 1)
 
    chi_stat, p_val = mcnemar_exact(b, c)
 
    return {
        "intersection_size": n_shared,
        "b": b,
        "c": c,
        "chi2":    round(chi_stat, 4) if not np.isnan(chi_stat) else None,
        "p_value": round(p_val, 6)    if not np.isnan(p_val)    else None,
        "significant_uncorrected": bool(p_val < 0.05)             if not np.isnan(p_val) else False,
        "significant_bonferroni":  bool(p_val < BONFERRONI_ALPHA) if not np.isnan(p_val) else False,
    }


significance_results = {}
mcnemar_results      = {}
mcnemar_available    = True 

for atk in ATTACKS:
    significance_results[atk] = {}
    for i, m1 in enumerate(MODEL_KEYS):
        for m2 in MODEL_KEYS[i+1:]:
            r1 = master[m1][atk]
            r2 = master[m2][atk]
 
            z_stat, p_val = two_proportion_ztest(
                r1["n_evaded"], r1["n_attacked"],
                r2["n_evaded"], r2["n_attacked"],
            )
 
            key = f"{MODEL_LABELS[m1]} vs {MODEL_LABELS[m2]}"
            significance_results[atk][key] = {
                "n1_evaded":   r1["n_evaded"],
                "n1_attacked": r1["n_attacked"],
                "asr1":        round(r1["n_evaded"] / max(r1["n_attacked"], 1), 6),
                "n2_evaded":   r2["n_evaded"],
                "n2_attacked": r2["n_attacked"],
                "asr2":        round(r2["n_evaded"] / max(r2["n_attacked"], 1), 6),
                "z_statistic": round(z_stat, 4) if not np.isnan(z_stat) else None,
                "p_value":     round(p_val, 6)  if not np.isnan(p_val)  else None,
                "significant_uncorrected": bool(p_val < 0.05)             if not np.isnan(p_val) else False,
                "significant_bonferroni":  bool(p_val < BONFERRONI_ALPHA) if not np.isnan(p_val) else False,
            }
 
    per_graph_data = _load_per_graph_for_attack(atk)
 
    if per_graph_data is None:
        mcnemar_available = False
        mcnemar_results[atk] = {
            "_status": "skipped — per_graph data not available"
        }
        continue
 
    mcnemar_results[atk] = {}
    for i, m1 in enumerate(MODEL_KEYS):
        for m2 in MODEL_KEYS[i+1:]:
            key = f"{MODEL_LABELS[m1]} vs {MODEL_LABELS[m2]}"
            mcnemar_results[atk][key] = compute_mcnemar_for_pair(
                per_graph_data[m1], per_graph_data[m2]
            )
 
if mcnemar_available:
    print("Statistical tests done (z-test + McNemar).")
else:
    print("Statistical tests done (z-test only; McNemar skipped — "
          "re-run attacks with per_graph saving to enable).")
 
def compute_transfer_adjusted_asr():
    results = {}
 
    for eps in sorted(transfer_raw["gcn_to_targets"].keys(), key=float):
        results[eps] = {}
        for m in MODEL_KEYS:
            t_data = transfer_raw["gcn_to_targets"][eps][m]
            t_pg = t_data.get("per_graph", {})
            t_indices = t_pg.get("subgraph_indices", [])
            t_evaded = t_pg.get("evaded", [])
            t_conf_pre = t_pg.get("confidence_pre", [])
 
            if not t_indices or not t_evaded:
                results[eps][m] = {"status": "per_graph data missing"}
                continue
 
            first_pgd_eps = list(pgd_raw["models"][m]["epsilon_results"].keys())[0]
            pgd_pg = pgd_raw["models"][m]["epsilon_results"][first_pgd_eps].get("per_graph", {})
            target_tp_indices = set(pgd_pg.get("subgraph_indices", []))
 
            if not target_tp_indices:
                results[eps][m] = {"status": "target TP indices not available"}
                continue
 
            transfer_map = dict(zip(t_indices, t_evaded))
            transfer_conf_map = dict(zip(t_indices, t_conf_pre)) if t_conf_pre else {}
 
            tp_in_transfer = set(t_indices) & target_tp_indices
            n_tp_target = len(tp_in_transfer)
 
            evaded_on_tp = sum(1 for idx in tp_in_transfer if transfer_map.get(idx, 0) == 1)
 
            all_evaded_indices = {idx for idx, ev in zip(t_indices, t_evaded) if ev == 1}
            vacuous = all_evaded_indices - target_tp_indices
            n_vacuous = len(vacuous)
 
            real_evaded = all_evaded_indices & target_tp_indices
            conf_real = [transfer_conf_map.get(idx, None) for idx in real_evaded
                         if transfer_conf_map.get(idx, None) is not None]
            conf_vacuous = [transfer_conf_map.get(idx, None) for idx in vacuous
                            if transfer_conf_map.get(idx, None) is not None]
 
            adjusted_asr = evaded_on_tp / n_tp_target if n_tp_target > 0 else 0.0
            raw_asr = t_data.get("transfer_asr", 0.0)
 
            results[eps][m] = {
                "n_gcn_tp": len(t_indices),
                "n_tp_target_in_pool": n_tp_target,
                "n_not_tp_target": len(t_indices) - n_tp_target,
                "n_evaded_total": int(sum(t_evaded)),
                "n_evaded_on_tp_target": evaded_on_tp,
                "n_vacuous_evasions": n_vacuous,
                "raw_asr": round(raw_asr, 6),
                "adjusted_asr": round(adjusted_asr, 6),
                "asr_difference_pp": round((raw_asr - adjusted_asr) * 100, 2),
                "mean_conf_real_evasions": round(float(np.mean(conf_real)), 6) if conf_real else None,
                "mean_conf_vacuous_evasions": round(float(np.mean(conf_vacuous)), 6) if conf_vacuous else None,
            }
 
    return results
 
 
def compute_cross_attack_overlap():
    results = {}
 
    for m in MODEL_KEYS:
        evaded_sets = {}
 
        for atk in ATTACKS:
            pg_data = None
            try:
                if atk == "pgd":
                    pg_data = pgd_raw["models"][m]["epsilon_results"][CANON_EPS].get("per_graph", {})
                elif atk == "zoo":
                    pg_data = zoo_raw["models"][m]["epsilon_results"][CANON_EPS].get("per_graph", {})
                elif atk == "transfer":
                    pg_data = transfer_raw["gcn_to_targets"][CANON_EPS][m].get("per_graph", {})
                elif atk == "saliency_grad":
                    pg_data = (sal_raw["models"][m]["topk_results"][CANON_SAL_K]
                               ["epsilon_results"][CANON_EPS].get("per_graph", {}))
                elif atk == "saliency_ig":
                    pg_data = (sal_ig_raw["models"][m]["topk_results"][CANON_SAL_K]
                               ["epsilon_results"][CANON_EPS].get("per_graph", {}))
                elif atk == "injection":
                    pg_data = inj_raw["models"][m]["combinations"][CANON_INJ].get("per_graph", {})
                elif atk == "rewiring":
                    pg_data = topo_raw["models"][m]["combinations"][CANON_TOPO].get("per_graph", {})
            except (KeyError, TypeError):
                continue
 
            if pg_data:
                indices = pg_data.get("subgraph_indices", [])
                evaded = pg_data.get("evaded", [])
                evaded_sets[atk] = {idx for idx, ev in zip(indices, evaded) if ev == 1}
            else:
                evaded_sets[atk] = set()
 
        overlap_attacks = [a for a in ATTACKS if a != "transfer"]
        overlap_sets = {a: evaded_sets.get(a, set()) for a in overlap_attacks}
 
        jaccard = {}
        for i, a1 in enumerate(overlap_attacks):
            for a2 in overlap_attacks[i+1:]:
                s1, s2 = overlap_sets[a1], overlap_sets[a2]
                union = s1 | s2
                inter = s1 & s2
                j = len(inter) / len(union) if union else 0.0
                jaccard[f"{ATTACK_SHORT[a1]} vs {ATTACK_SHORT[a2]}"] = {
                    "jaccard": round(j, 4),
                    "intersection": len(inter),
                    "union": len(union),
                }
 
        non_empty = [s for s in overlap_sets.values() if s]
        if non_empty:
            core = set.intersection(*non_empty)
        else:
            core = set()
 
        any_evaded = set.union(*non_empty) if non_empty else set()
 
        unique = {}
        for atk in overlap_attacks:
            others = set.union(*[overlap_sets[a] for a in overlap_attacks if a != atk]) if len(overlap_attacks) > 1 else set()
            unique[ATTACK_SHORT[atk]] = len(overlap_sets[atk] - others)
 
        first_pgd_eps = list(pgd_raw["models"][m]["epsilon_results"].keys())[0]
        n_tp = pgd_raw["models"][m]["epsilon_results"][first_pgd_eps].get("n_attacked", 0)
 
        results[m] = {
            "n_tp": n_tp,
            "n_core_vulnerable": len(core),
            "core_vulnerable_pct": round(len(core) / n_tp * 100, 2) if n_tp > 0 else 0.0,
            "n_any_evaded": len(any_evaded),
            "any_evaded_pct": round(len(any_evaded) / n_tp * 100, 2) if n_tp > 0 else 0.0,
            "n_never_evaded": n_tp - len(any_evaded),
            "never_evaded_pct": round((n_tp - len(any_evaded)) / n_tp * 100, 2) if n_tp > 0 else 0.0,
            "unique_evasions_per_attack": unique,
            "pairwise_jaccard": jaccard,
        }
 
    return results
 
 
transfer_adjusted = compute_transfer_adjusted_asr()
cross_attack_overlap = compute_cross_attack_overlap()
print("Transfer adjusted ASR and cross-attack overlap computed.")


def bootstrap_ci_asr(n_evaded, n_attacked, n_boot=10_000, alpha=0.05, rng_seed=42):
    if n_attacked == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(rng_seed)
    obs = np.zeros(n_attacked, dtype=int)
    obs[:n_evaded] = 1
    boot_asrs = np.array([
        rng.choice(obs, size=n_attacked, replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(boot_asrs, 100 * alpha / 2))
    hi = float(np.percentile(boot_asrs, 100 * (1 - alpha / 2)))
    return lo, hi
 
 
ci_results = {}
for m in MODEL_KEYS:
    ci_results[m] = {}
    for atk in ATTACKS:
        r = master[m][atk]
        lo, hi = bootstrap_ci_asr(r["n_evaded"], r["n_attacked"])
        ci_results[m][atk] = {"lo": round(lo, 4), "hi": round(hi, 4)}
 

FEATURE_ATTACKS    = ["pgd", "zoo", "transfer", "saliency_grad", "saliency_ig"]
STRUCTURAL_ATTACKS = ["injection", "rewiring"]

mean_asr          = {m: np.mean([master[m][a]["asr"] for a in ATTACKS]) for m in MODEL_KEYS}
mean_asr_feature  = {m: np.mean([master[m][a]["asr"] for a in FEATURE_ATTACKS]) for m in MODEL_KEYS}
mean_asr_struct   = {m: np.mean([master[m][a]["asr"] for a in STRUCTURAL_ATTACKS]) for m in MODEL_KEYS}

robustness_rank = sorted(MODEL_KEYS, key=lambda m: mean_asr[m])

mean_per_attack = {a: np.mean([master[m][a]["asr"] for m in MODEL_KEYS]) for a in ATTACKS}
attack_rank     = sorted(ATTACKS, key=lambda a: mean_per_attack[a], reverse=True)

print("Robustness ranking (most robust first):",
      [MODEL_LABELS[m] for m in robustness_rank])
print("Attack ranking (most effective first):",
      [ATTACK_LABELS[a] for a in attack_rank])


comparison_out = {
    "experiment": "final_comparison",
    "seed": 42,
    "canonical_configs": {
        "feature_attacks_epsilon": CANON_EPS,
        "saliency_top_k": CANON_SAL_K,
        "injection_combo": CANON_INJ,
        "topology_combo": CANON_TOPO,
    },
    "statistical_test": {
        "primary": {
            "method": "two-proportion z-test (continuity corrected)",
            "note": ("Compares independent ASR proportions. Valid even when "
                     "models have different TP pools (different subgraphs attacked)."),
        },
        "supplementary": {
            "method": "McNemar's chi-squared test (continuity corrected)",
            "note": ("Paired test on the intersection of TP subgraphs. "
                     "More powerful but limited to shared subgraphs. "
                     "Requires per_graph data from attack scripts."),
            "available": mcnemar_available,
        },
        "bonferroni_alpha": round(BONFERRONI_ALPHA, 6),
        "n_comparisons_per_attack": N_MODEL_PAIRS,
    },
    "significance_tests_zprop": significance_results,
    "significance_tests_mcnemar": mcnemar_results,
    "models": {
        m: {
            "display_name": MODEL_LABELS[m],
            "baseline_f1_macro": baseline_raw[MODEL_LABELS[m]]["f1_macro"]["mean"],
            "baseline_f1_std": baseline_raw[MODEL_LABELS[m]]["f1_macro"]["std"],
            "mean_asr_all":      round(mean_asr[m], 4),
            "mean_asr_feature":  round(mean_asr_feature[m], 4),
            "mean_asr_struct":   round(mean_asr_struct[m], 4),
            "robustness_rank": robustness_rank.index(m) + 1,
            "attacks": {
                atk: {
                    "asr":        round(master[m][atk]["asr"], 4),
                    "f1_drop":    round(master[m][atk]["f1_drop"], 4),
                    "budget":     master[m][atk]["budget"],
                    "n_evaded":   master[m][atk]["n_evaded"],
                    "n_attacked": master[m][atk]["n_attacked"],
                    "asr_ci_lo":  ci_results[m][atk]["lo"],
                    "asr_ci_hi":  ci_results[m][atk]["hi"],
                    "max_asr":    round(max_asr[m][atk], 4),
                }
                for atk in ATTACKS
            },
        }
        for m in MODEL_KEYS
    },
    "attack_ranking": {
        a: {
            "label": ATTACK_LABELS[a],
            "mean_asr": round(mean_per_attack[a], 4),
            "rank": attack_rank.index(a) + 1,
        }
        for a in ATTACKS
    },
    "bootstrap_ci": ci_results,
    "robustness_ranking_most_to_least_robust": [MODEL_LABELS[m] for m in robustness_rank],
    "transfer_adjusted_asr": transfer_adjusted,
    "cross_attack_overlap": cross_attack_overlap,
}

out_path = RESULTS / "final_comparison.json"
with open(out_path, "w") as f:
    json.dump(comparison_out, f, indent=2)
print(f"Saved {out_path}")


def radar_chart():
    categories = [ATTACK_SHORT[a] for a in ATTACKS]
    N = len(categories)
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10),
                             subplot_kw=dict(polar=True))
    axes = axes.flatten()

    for idx, m in enumerate(MODEL_KEYS):
        ax = axes[idx]
        values = [max_asr[m][a] for a in ATTACKS]
        values += values[:1]

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(["20%", "40%", "60%", "80%"], size=8, color="grey")

        color = MODEL_COLORS[m]
        ax.plot(angles, values, "o-", linewidth=2, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
        ax.set_title(MODEL_LABELS[m], size=13, fontweight="bold",
                     pad=15, color=color)

        for angle, val, cat in zip(angles[:-1], values[:-1], categories):
            ax.annotate(f"{val*100:.0f}%",
                        xy=(angle, val),
                        xytext=(angle, val + 0.07),
                        ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    fig.suptitle("Adversarial Robustness Profile — Maximum ASR per Attack Type\n"
                 "(worst-case across all configurations)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = PLOTS / "final_radar.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

radar_chart()

def asr_vs_budget():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    feat_attacks = {
        "pgd":           {"eps": [0.01, 0.05, 0.1, 0.2, 0.5]},
        "zoo":           {"eps": [0.05, 0.1, 0.2]},
        "transfer":      {"eps": [0.05, 0.1, 0.2]},
        "saliency_grad": {"eps": [0.1, 0.2], "topk": "5"},
        "saliency_ig":   {"eps": [0.1, 0.2], "topk": "5"},
    }

    linestyles = {"gatv2": "-", "sage": "--", "sage_edge": "-.", "gin": ":"}
    markers    = {"gatv2": "o", "sage": "s", "sage_edge": "^", "gin": "D"}

    def _get_asr(atk, m, eps_str, topk="5"):
        if atk == "pgd":
            return pgd_raw["models"][m]["epsilon_results"][eps_str]["asr"]
        elif atk == "zoo":
            return zoo_raw["models"][m]["epsilon_results"][eps_str]["asr"]
        elif atk == "transfer":
            return transfer_raw["gcn_to_targets"][eps_str][m]["transfer_asr"]
        elif atk == "saliency_grad":
            return sal_raw["models"][m]["topk_results"][topk]["epsilon_results"][eps_str]["asr"]
        elif atk == "saliency_ig":
            return sal_ig_raw["models"][m]["topk_results"][topk]["epsilon_results"][eps_str]["asr"]

    for atk, cfg in feat_attacks.items():
        topk = cfg.get("topk", "5")
        for m in MODEL_KEYS:
            asrs = [_get_asr(atk, m, str(eps), topk) for eps in cfg["eps"]]
            label = f"{MODEL_LABELS[m]}" if atk == "pgd" else None
            ax1.plot(cfg["eps"], [a * 100 for a in asrs],
                     color=MODEL_COLORS[m],
                     linestyle=linestyles[m],
                     marker=markers[m],
                     linewidth=2, markersize=6,
                     label=label, alpha=0.85)

        avg_last = np.mean([
            _get_asr(atk, m, str(cfg["eps"][-1]), cfg.get("topk", "5"))
            for m in MODEL_KEYS
        ]) * 100
        ax1.annotate(ATTACK_SHORT[atk],
                     xy=(cfg["eps"][-1], avg_last),
                     xytext=(cfg["eps"][-1] + 0.01, avg_last),
                     fontsize=9, color=ATTACK_COLORS[atk], fontweight="bold",
                     va="center")

    ax1.set_xlabel(r"Perturbation Budget $\varepsilon$ (L$_\infty$)", fontsize=12)
    ax1.set_ylabel("Attack Success Rate (%)", fontsize=12)
    ax1.set_title("Feature-Space Attacks: ASR vs $\\varepsilon$", fontsize=13)
    ax1.legend(title="Model", loc="upper left", framealpha=0.9)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    topo_k_vals = [2, 5, 10]
    inj_k_vals  = [1, 3, 5, 10]

    for m in MODEL_KEYS:
        asrs = [topo_raw["models"][m]["combinations"][f"chain_injection_k{k}"]["asr"]
                for k in topo_k_vals]
        ax2.plot(topo_k_vals, [a * 100 for a in asrs],
                 color=MODEL_COLORS[m],
                 linestyle="-", marker="o",
                 linewidth=2, markersize=7,
                 label=f"{MODEL_LABELS[m]} (Rewire)", alpha=0.9)

        asrs_inj = [inj_raw["models"][m]["combinations"][f"random_k{k}_c1"]["asr"]
                    for k in inj_k_vals]
        ax2.plot(inj_k_vals, [a * 100 for a in asrs_inj],
                 color=MODEL_COLORS[m],
                 linestyle=":", marker="^",
                 linewidth=2, markersize=7,
                 label=f"{MODEL_LABELS[m]} (Inject)", alpha=0.6)

    model_handles = [
        mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m])
        for m in MODEL_KEYS
    ]
    style_handles = [
        plt.Line2D([0], [0], linestyle="-", marker="o", color="grey",
                   label="Topology Rewiring (peel-chain)"),
        plt.Line2D([0], [0], linestyle=":", marker="^", color="grey",
                   label="Node Injection (random, c=1)"),
    ]
    ax2.legend(handles=model_handles + style_handles,
               loc="upper left", fontsize=9, framealpha=0.9)

    ax2.set_xlabel("Number of Injected / Intermediate Nodes k", fontsize=12)
    ax2.set_ylabel("Attack Success Rate (%)", fontsize=12)
    ax2.set_title("Topology Attacks: ASR vs Structural Budget", fontsize=13)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    ax2.set_xticks([1, 2, 3, 5, 10])

    fig.suptitle("ASR vs Perturbation Budget — All Attack Types", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    path = PLOTS / "final_asr_vs_budget.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

asr_vs_budget()

def print_transfer_adjusted():
    hdr("11. TRANSFER ADJUSTED ASR  (evaded ∩ TP_target / TP_target)")
    print("  Raw ASR counts all GCN-surrogate TP as candidates.")
    print("  Adjusted ASR counts only subgraphs that the TARGET model also classifies as suspicious.\n")
 
    for eps in sorted(transfer_adjusted.keys(), key=float):
        print(f"  ε = {eps}")
        print(f"    {'Model':<12} {'GCN-TP':>7} {'TP_tgt':>7} {'NotTP':>6} "
              f"{'Evad_all':>9} {'Evad_TP':>8} {'Vacuous':>8} "
              f"{'Raw ASR':>9} {'Adj ASR':>9} {'Δpp':>6} "
              f"{'Conf_real':>10} {'Conf_vac':>10}")
        sep("-")
        for m in MODEL_KEYS:
            r = transfer_adjusted[eps].get(m, {})
            if "status" in r:
                print(f"    {MODEL_LABELS[m]:<12} {r['status']}")
                continue
            conf_r = f"{r['mean_conf_real_evasions']:.6f}" if r['mean_conf_real_evasions'] is not None else "n/a"
            conf_v = f"{r['mean_conf_vacuous_evasions']:.6f}" if r['mean_conf_vacuous_evasions'] is not None else "n/a"
            print(f"    {MODEL_LABELS[m]:<12} "
                  f"{r['n_gcn_tp']:>7} "
                  f"{r['n_tp_target_in_pool']:>7} "
                  f"{r['n_not_tp_target']:>6} "
                  f"{r['n_evaded_total']:>9} "
                  f"{r['n_evaded_on_tp_target']:>8} "
                  f"{r['n_vacuous_evasions']:>8} "
                  f"{r['raw_asr']*100:>8.1f}% "
                  f"{r['adjusted_asr']*100:>8.1f}% "
                  f"{r['asr_difference_pp']:>5.1f} "
                  f"{conf_r:>10} "
                  f"{conf_v:>10}")
        print()
 
 
def print_cross_attack_overlap():
    hdr("12. CROSS-ATTACK OVERLAP  (canonical configs, transfer excluded)")
    print("  Core vulnerable = evaded by ALL attacks. Never evaded = robust to ALL attacks.\n")
 
    print(f"  {'Model':<12} {'n_TP':>6} {'Core':>6} {'Core%':>7} "
          f"{'Any':>6} {'Any%':>7} {'Never':>7} {'Never%':>8}")
    sep("-")
    for m in MODEL_KEYS:
        r = cross_attack_overlap[m]
        print(f"  {MODEL_LABELS[m]:<12} "
              f"{r['n_tp']:>6} "
              f"{r['n_core_vulnerable']:>6} "
              f"{r['core_vulnerable_pct']:>6.1f}% "
              f"{r['n_any_evaded']:>6} "
              f"{r['any_evaded_pct']:>6.1f}% "
              f"{r['n_never_evaded']:>7} "
              f"{r['never_evaded_pct']:>7.1f}%")
 
    print(f"\n  Unique evasions (evaded ONLY by this attack, not by any other):")
    print(f"  {'Model':<12}", end="")
    first_m = MODEL_KEYS[0]
    atk_names = list(cross_attack_overlap[first_m]["unique_evasions_per_attack"].keys())
    for a in atk_names:
        print(f" {a:>10}", end="")
    print()
    sep("-")
    for m in MODEL_KEYS:
        r = cross_attack_overlap[m]
        print(f"  {MODEL_LABELS[m]:<12}", end="")
        for a in atk_names:
            print(f" {r['unique_evasions_per_attack'].get(a, 0):>10}", end="")
        print()
 
    print(f"\n  Pairwise Jaccard similarity (selected pairs):")
    for m in MODEL_KEYS:
        jaccards = cross_attack_overlap[m]["pairwise_jaccard"]
        sorted_j = sorted(jaccards.items(), key=lambda x: x[1]["jaccard"], reverse=True)
        print(f"\n  {MODEL_LABELS[m]}:")
        print(f"    {'Pair':<30} {'Jaccard':>8} {'∩':>5} {'∪':>5}")
        for pair, vals in sorted_j:
            print(f"    {pair:<30} {vals['jaccard']:>8.4f} {vals['intersection']:>5} {vals['union']:>5}")


def feature_heatmap():
    top_n = 10
    feat_rank = {}
    for m in MODEL_KEYS:
        scores = np.array(sal_raw["models"][m]["feature_importance"]["gradient_scores"])
        ranked = np.argsort(scores)[::-1]
        feat_rank[m] = ranked[:top_n].tolist()

    ig_rank = {}
    for m in MODEL_KEYS:
        scores = np.array(sal_raw["models"][m]["feature_importance"]["ig_scores"])
        ranked = np.argsort(scores)[::-1]
        ig_rank[m] = ranked[:top_n].tolist()

    n_feats = 43
    presence = np.zeros((n_feats, len(MODEL_KEYS)), dtype=float)
    for j, m in enumerate(MODEL_KEYS):
        for rank_pos, feat_idx in enumerate(feat_rank[m]):
            presence[feat_idx, j] = 1.0 - rank_pos / top_n

    total_presence = presence.sum(axis=1)
    sort_idx = np.argsort(total_presence)[::-1][:25]
    pres_sub = presence[sort_idx, :]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    im = ax1.imshow(pres_sub.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax1.set_xticks(range(len(sort_idx)))
    ax1.set_xticklabels([f"F{i}" for i in sort_idx], rotation=90, fontsize=9)
    ax1.set_yticks(range(len(MODEL_KEYS)))
    ax1.set_yticklabels([MODEL_LABELS[m] for m in MODEL_KEYS], fontsize=11)
    ax1.set_title("Gradient-Based Feature Importance\n(Top-10 per model, rank-weighted)",
                  fontsize=12)
    ax1.set_xlabel("Feature index", fontsize=11)
    plt.colorbar(im, ax=ax1, label="Rank-weighted importance")

    for j, m in enumerate(MODEL_KEYS):
        for rank_pos, feat_idx in enumerate(feat_rank[m][:3]):
            if feat_idx in sort_idx:
                xi = np.where(sort_idx == feat_idx)[0][0]
                ax1.text(xi, j, "*", ha="center", va="center",
                         fontsize=14, color="black", fontweight="bold")

    def jaccard(a, b):
        return len(a & b) / len(a | b) if (a | b) else 0.0

    top_sets_grad = {m: set(feat_rank[m]) for m in MODEL_KEYS}
    J_grad = np.array([[jaccard(top_sets_grad[m1], top_sets_grad[m2])
                        for m2 in MODEL_KEYS] for m1 in MODEL_KEYS])

    labels = [MODEL_LABELS[m] for m in MODEL_KEYS]
    im2 = ax2.imshow(J_grad, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            v = J_grad[i, j]
            color = "white" if v > 0.6 else "black"
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=13, fontweight="bold", color=color)

    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
    ax2.set_yticklabels(labels, fontsize=11)
    ax2.set_title("Jaccard Similarity of Top-10 Important Features\n(Gradient method)",
                  fontsize=12)
    plt.colorbar(im2, ax=ax2, label="Jaccard similarity")

    fig.suptitle("Feature Importance Consensus Across GNN Models",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = PLOTS / "final_feature_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

feature_heatmap()


def domain_vs_standard():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    x   = np.arange(len(MODEL_KEYS))
    n_a = len(ATTACKS)
    w   = 0.11
    offsets = np.linspace(-(n_a - 1) * w / 2, (n_a - 1) * w / 2, n_a)

    ax = axes[0]
    for i, atk in enumerate(ATTACKS):
        asrs = [master[m][atk]["asr"] * 100 for m in MODEL_KEYS]
        ci_lo = [(master[m][atk]["asr"] - ci_results[m][atk]["lo"]) * 100
                 for m in MODEL_KEYS]
        ci_hi = [(ci_results[m][atk]["hi"] - master[m][atk]["asr"]) * 100
                 for m in MODEL_KEYS]
        ax.bar(x + offsets[i], asrs, width=w,
               color=ATTACK_COLORS[atk], label=ATTACK_LABELS[atk],
               alpha=0.88, edgecolor="white", linewidth=0.5)
        ax.errorbar(x + offsets[i], asrs,
                    yerr=[ci_lo, ci_hi],
                    fmt="none", color="black", capsize=2, linewidth=0.8)

    domain_mean = [mean_asr_struct[m] * 100 for m in MODEL_KEYS]
    std_mean    = [mean_asr_feature[m] * 100 for m in MODEL_KEYS]

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_KEYS], fontsize=12)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
    ax.set_title("ASR Comparison: All Attack Methods per Model", fontsize=13)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    for xi, (dm, sm) in enumerate(zip(domain_mean, std_mean)):
        ratio = dm / sm if sm > 0 else float("inf")
        ax.annotate(f"S:{sm:.0f}%\nD:{dm:.0f}%\n({ratio:.1f}x)",
                    xy=(xi, 102), ha="center", va="top",
                    fontsize=7, color="grey")

    ax2 = axes[1]
    cat_colors = ["#5b9bd5", "#ed7d31"]
    w2 = 0.35
    for xi, m in enumerate(MODEL_KEYS):
        std_f1 = np.mean([master[m][a]["f1_drop"] for a in FEATURE_ATTACKS])
        dom_f1 = np.mean([master[m][a]["f1_drop"] for a in STRUCTURAL_ATTACKS])

        ax2.bar(xi - w2 / 2, std_f1, width=w2, color=cat_colors[0],
                label="Feature-space" if xi == 0 else None, alpha=0.85, edgecolor="white")
        ax2.bar(xi + w2 / 2, dom_f1, width=w2, color=cat_colors[1],
                label="Domain-specific" if xi == 0 else None, alpha=0.85, edgecolor="white")

        if dom_f1 > std_f1 * 1.5:
            ax2.text(xi, max(std_f1, dom_f1) + 0.005, "***",
                     ha="center", fontsize=14, color="black")
        elif dom_f1 > std_f1 * 1.2:
            ax2.text(xi, max(std_f1, dom_f1) + 0.005, "*",
                     ha="center", fontsize=12, color="black")

    ax2.set_xticks(range(len(MODEL_KEYS)))
    ax2.set_xticklabels([MODEL_LABELS[m] for m in MODEL_KEYS], fontsize=12)
    ax2.set_ylabel("Mean F1-macro Drop", fontsize=12)
    ax2.set_title("Mean F1 Drop: Domain-specific vs Feature-space Attacks", fontsize=13)
    ax2.legend(fontsize=11, framealpha=0.9)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Domain-Specific vs Standard Adversarial Attacks on Bitcoin AML GNNs",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = PLOTS / "final_domain_vs_standard.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

domain_vs_standard()


def significance_heatmap():
    n_atks = len(ATTACKS)
    ncols = 4
    nrows = (n_atks + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
    axes = axes.flatten()
    for idx in range(n_atks, len(axes)):
        axes[idx].set_visible(False)

    for ax_idx, atk in enumerate(ATTACKS):
        ax = axes[ax_idx]
        p_mat   = np.ones((4, 4))
        sig_mat = np.zeros((4, 4), dtype=int)

        for i, m1 in enumerate(MODEL_KEYS):
            for j, m2 in enumerate(MODEL_KEYS):
                if i == j:
                    continue
                pair_key = (f"{MODEL_LABELS[m1]} vs {MODEL_LABELS[m2]}"
                            if i < j else
                            f"{MODEL_LABELS[m2]} vs {MODEL_LABELS[m1]}")
                res = significance_results[atk].get(pair_key, {})
                p_val = res.get("p_value", 1.0)
                p_mat[i, j] = p_val if p_val is not None else 1.0

                if res.get("significant_bonferroni"):
                    sig_mat[i, j] = 2
                elif res.get("significant_uncorrected"):
                    sig_mat[i, j] = 1

        im = ax.imshow(p_mat, cmap="RdYlGn_r", vmin=0, vmax=0.1)
        for i in range(4):
            for j in range(4):
                if i == j:
                    txt = "--"
                    marker = ""
                else:
                    txt = f"{p_mat[i,j]:.4f}"
                    marker = " *" if sig_mat[i, j] == 2 else (" †" if sig_mat[i, j] == 1 else "")
                color = "white" if p_mat[i, j] < 0.02 else "black"
                ax.text(j, i, txt + marker, ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

        labels = [MODEL_LABELS[m] for m in MODEL_KEYS]
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_title(f"{ATTACK_SHORT[atk]}", fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Pairwise Model Comparison: Two-Proportion Z-Test p-values\n"
                 f"(* = Bonferroni-significant at α={BONFERRONI_ALPHA:.4f}, "
                 f"† = significant at α=0.05)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = PLOTS / "final_significance_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

significance_heatmap()


W = 90

def sep(char="=", width=W):
    print(char * width)

def hdr(title, char="═", width=W):
    pad = max(0, width - len(title) - 4)
    print(f"\n{char*2} {title} {char*(pad)}")

sep()
print("COMPREHENSIVE ADVERSARIAL ATTACK ANALYSIS — Elliptic2 Bitcoin AML")
print(f"  7 attack methods  x  4 GNN models  |  canonical ε={CANON_EPS}, "
      f"top-k={CANON_SAL_K}, inj={CANON_INJ}, topo={CANON_TOPO}")
sep()

hdr("0. BASELINE MODEL PERFORMANCE (clean, multi-seed)")
print(f"  {'Model':<14} {'F1-macro mean':>14} {'F1-macro std':>13} "
      f"{'F1-macro min':>13} {'F1-macro max':>13}")
sep("-")
for m in MODEL_KEYS:
    b = baseline_raw[MODEL_LABELS[m]]["f1_macro"]
    mn  = b.get("min", float("nan"))
    mx  = b.get("max", float("nan"))
    print(f"  {MODEL_LABELS[m]:<14} {b['mean']:>14.4f} {b['std']:>13.4f} "
          f"{mn:>13.4f} {mx:>13.4f}")

hdr("1. CANONICAL RESULTS TABLE  (ASR / ΔF1 / n_evaded/n_attacked / conf_pre→post)")
col = f"  {'Attack':<18}"
for m in MODEL_KEYS:
    col += f" {MODEL_LABELS[m]:>21}"
print(col)
sep("-")
for atk in ATTACKS:
    row  = f"  {ATTACK_LABELS[atk]:<18}"
    for m in MODEL_KEYS:
        r   = master[m][atk]
        ci  = ci_results[m][atk]
        row += (f" {r['asr']*100:5.1f}%"
                f"[{ci['lo']*100:.1f}-{ci['hi']*100:.1f}]"
                f" Δ{r['f1_drop']:.6f}")
    print(row)
    row2 = f"  {'  n_evaded/attacked':<18}"
    for m in MODEL_KEYS:
        r = master[m][atk]
        row2 += f" {r['n_evaded']:>6}/{r['n_attacked']:<6}           "
    print(row2)
    row3 = f"  {'  conf pre→post':<18}"
    for m in MODEL_KEYS:
        r = master[m][atk]
        row3 += f" {r['conf_pre']:.6f}→{r['conf_post']:.6f}              "
    print(row3)
    if atk == "zoo":
        row4 = f"  {'  mean queries':<18}"
        for m in MODEL_KEYS:
            q = zoo_raw["models"][m]["epsilon_results"][CANON_EPS].get("mean_queries_used", float("nan"))
            row4 += f" {q:>8.1f}                 "
        print(row4)
    sep("-")

hdr("2. MAX ASR TABLE  (best configuration across all budgets / combos)")
row = f"  {'Attack':<18}"
for m in MODEL_KEYS:
    row += f" {MODEL_LABELS[m]:>10}"
row += f"  {'mean':>8}"
print(row)
sep("-")
for atk in ATTACKS:
    row = f"  {ATTACK_LABELS[atk]:<18}"
    asrs = [max_asr[m][atk] for m in MODEL_KEYS]
    for a in asrs:
        row += f" {a*100:9.1f}%"
    row += f"  {np.mean(asrs)*100:7.1f}%"
    print(row)

hdr("3. FULL SWEEP RESULTS")

print("\n  [PGD — white-box, all epsilons]")
pgd_eps = sorted(pgd_raw["models"][MODEL_KEYS[0]]["epsilon_results"].keys(),
                 key=float)
hrow = f"  {'eps':>6}"
for m in MODEL_KEYS:
    hrow += f" {MODEL_LABELS[m]:>14}"
print(hrow)
for eps in pgd_eps:
    row = f"  {eps:>6}"
    for m in MODEL_KEYS:
        r = pgd_raw["models"][m]["epsilon_results"][eps]
        row += f" {r['asr']*100:6.1f}% Δ{abs(r['clean_f1_macro']-r['post_f1_macro']):.6f}"
    print(row)

print("\n  [ZOO — black-box, all epsilons]")
zoo_eps = sorted(zoo_raw["models"][MODEL_KEYS[0]]["epsilon_results"].keys(),
                 key=float)
hrow = f"  {'eps':>6}"
for m in MODEL_KEYS:
    hrow += f" {MODEL_LABELS[m]:>22}"
print(hrow)
for eps in zoo_eps:
    row = f"  {eps:>6}"
    for m in MODEL_KEYS:
        r = zoo_raw["models"][m]["epsilon_results"][eps]
        q = r.get("mean_queries_used", float("nan"))
        row += f" {r['asr']*100:5.1f}% Δ{abs(r['clean_f1_macro']-r['post_f1_macro']):.6f} q{q:.0f}"
    print(row)

print("\n  [Transfer (GCN→target) — gray-box, all epsilons]")
trans_eps = sorted(transfer_raw["gcn_to_targets"].keys(), key=float)
hrow = f"  {'eps':>6}"
for m in MODEL_KEYS:
    hrow += f" {MODEL_LABELS[m]:>14}"
print(hrow)
for eps in trans_eps:
    row = f"  {eps:>6}"
    for m in MODEL_KEYS:
        r = transfer_raw["gcn_to_targets"][eps][m]
        row += f" {r['transfer_asr']*100:6.1f}% Δ{abs(r['clean_f1_macro']-r['post_f1_macro']):.6f}"
    print(row)

print("\n  [Saliency-Grad — all top-k x epsilon combos]")
sal_topks = sorted(sal_raw["models"][MODEL_KEYS[0]]["topk_results"].keys(), key=int)
sal_eps   = sorted(
    sal_raw["models"][MODEL_KEYS[0]]["topk_results"][sal_topks[0]]["epsilon_results"].keys(),
    key=float)
for tk in sal_topks:
    print(f"    top-k={tk}:")
    hrow = f"      {'eps':>6}"
    for m in MODEL_KEYS:
        hrow += f" {MODEL_LABELS[m]:>14}"
    print(hrow)
    for eps in sal_eps:
        row = f"      {eps:>6}"
        for m in MODEL_KEYS:
            r = sal_raw["models"][m]["topk_results"][tk]["epsilon_results"][eps]
            row += f" {r['asr']*100:6.1f}% Δ{abs(r['clean_f1_macro']-r['post_f1_macro']):.6f}"
        print(row)

print("\n  [Saliency-IG — all top-k x epsilon combos]")
sal_ig_topks = sorted(sal_ig_raw["models"][MODEL_KEYS[0]]["topk_results"].keys(), key=int)
sal_ig_eps   = sorted(
    sal_ig_raw["models"][MODEL_KEYS[0]]["topk_results"][sal_ig_topks[0]]["epsilon_results"].keys(),
    key=float)
for tk in sal_ig_topks:
    print(f"    top-k={tk}:")
    hrow = f"      {'eps':>6}"
    for m in MODEL_KEYS:
        hrow += f" {MODEL_LABELS[m]:>14}"
    print(hrow)
    for eps in sal_ig_eps:
        row = f"      {eps:>6}"
        for m in MODEL_KEYS:
            r = sal_ig_raw["models"][m]["topk_results"][tk]["epsilon_results"][eps]
            row += f" {r['asr']*100:6.1f}% Δ{abs(r['clean_f1_macro']-r['post_f1_macro']):.6f}"
        print(row)

print("\n  [Node Injection — all combinations (k_nodes x connections)]")
inj_combos = sorted(inj_raw["models"][MODEL_KEYS[0]]["combinations"].keys())
hrow = f"  {'combo':<22}"
for m in MODEL_KEYS:
    hrow += f" {MODEL_LABELS[m]:>14}"
print(hrow)
for combo in inj_combos:
    row = f"  {combo:<22}"
    for m in MODEL_KEYS:
        r = inj_raw["models"][m]["combinations"][combo]
        row += f" {r['asr']*100:6.1f}% Δ{abs(r['clean_f1_macro']-r['post_f1_macro']):.6f}"
    print(row)

print("\n  [Topology Rewiring — all combinations]")
topo_combos = sorted(topo_raw["models"][MODEL_KEYS[0]]["combinations"].keys())
hrow = f"  {'combo':<22}"
for m in MODEL_KEYS:
    hrow += f" {MODEL_LABELS[m]:>14}"
print(hrow)
for combo in topo_combos:
    row = f"  {combo:<22}"
    for m in MODEL_KEYS:
        r = topo_raw["models"][m]["combinations"][combo]
        row += f" {r['asr']*100:6.1f}% Δ{abs(r['clean_f1_macro']-r['post_f1_macro']):.6f}"
    print(row)

hdr("4. ROBUSTNESS RANKING  (most → least robust)")
print(f"  {'Rank':<5} {'Model':<12} {'Baseline F1':>12} "
      f"{'Mean ASR (feat)':>17} {'Mean ASR (struct)':>18} "
      f"{'Mean ASR (all)':>15} {'Worst attack':<28} {'Max ASR':>8}")
sep("-")
for rank, m in enumerate(robustness_rank, 1):
    b       = baseline_raw[MODEL_LABELS[m]]["f1_macro"]["mean"]
    worst   = max(ATTACKS, key=lambda a: master[m][a]["asr"])
    worst_m = max_asr[m][worst]
    print(f"  {rank:<5} {MODEL_LABELS[m]:<12} {b:>12.4f} "
          f"{mean_asr_feature[m]*100:>16.1f}% "
          f"{mean_asr_struct[m]*100:>17.1f}% "
          f"{mean_asr[m]*100:>14.1f}% "
          f"{ATTACK_LABELS[worst]:<28} {worst_m*100:>7.1f}%")

hdr("5. ATTACK EFFECTIVENESS RANKING  (most → least effective)")
print(f"  {'Rank':<5} {'Attack':<26} {'Mean ASR':>9} "
      + "  ".join(f"{MODEL_LABELS[m]:>9}" for m in MODEL_KEYS))
sep("-")
for rank, atk in enumerate(attack_rank, 1):
    per_model = "  ".join(f"{master[m][atk]['asr']*100:8.1f}%" for m in MODEL_KEYS)
    print(f"  {rank:<5} {ATTACK_LABELS[atk]:<26} {mean_per_attack[atk]*100:8.1f}%  {per_model}")

hdr("6. DOMAIN-SPECIFIC vs FEATURE-SPACE ATTACKS")
print(f"  {'Model':<14} {'Feature ASR':>12} {'Structural ASR':>15} "
      f"{'Ratio (S/F)':>12} {'Feature ΔF1':>12} {'Structural ΔF1':>15}")
sep("-")
for m in MODEL_KEYS:
    std   = mean_asr_feature[m] * 100
    dom   = mean_asr_struct[m] * 100
    ratio = dom / std if std > 0 else float("inf")
    f1_f  = np.mean([master[m][a]["f1_drop"] for a in FEATURE_ATTACKS])
    f1_s  = np.mean([master[m][a]["f1_drop"] for a in STRUCTURAL_ATTACKS])
    print(f"  {MODEL_LABELS[m]:<14} {std:>11.1f}% {dom:>14.1f}% "
          f"{ratio:>12.2f}x {f1_f:>12.4f} {f1_s:>15.4f}")

hdr("7. CONFIDENCE ANALYSIS  (mean confidence pre → post attack, canonical config)")
print(f"  {'Attack':<18} {'Model':<12} {'Conf pre':>10} {'Conf post':>10} {'Drop':>8}")
sep("-")
for atk in ATTACKS:
    for m in MODEL_KEYS:
        r = master[m][atk]
        drop = r["conf_pre"] - r["conf_post"]
        print(f"  {ATTACK_LABELS[atk]:<18} {MODEL_LABELS[m]:<12} "
              f"{r['conf_pre']:>10.4f} {r['conf_post']:>10.4f} {drop:>8.4f}")
    sep("-")

hdr(f"8. TWO-PROPORTION Z-TEST — ALL PAIRS  "
    f"(Bonferroni α_B={BONFERRONI_ALPHA:.4f}, {N_MODEL_PAIRS} comparisons/attack)")
pairs = list(significance_results[ATTACKS[0]].keys())
print(f"  {'Model pair':<28} {'Attack':<18} "
      f"{'ASR-1':>7} {'n1':>6} {'ASR-2':>7} {'n2':>6} "
      f"{'z':>7} {'p':>9} {'sig':>6}")
sep("-")
n_sig_bonf_z = 0
n_sig_uncorr_z = 0
total_tests = N_MODEL_PAIRS * len(ATTACKS)
for atk in ATTACKS:
    for pair, res in significance_results[atk].items():
        p   = res.get("p_value")
        z   = res.get("z_statistic")
        if res.get("significant_bonferroni"):
            sig = " * "
            n_sig_bonf_z += 1
        elif res.get("significant_uncorrected"):
            sig = " † "
            n_sig_uncorr_z += 1
        else:
            sig = "   "
        p_str = f"{p:.5f}" if p is not None else "  n/a  "
        z_str = f"{z:.6f}"  if z is not None else "  n/a"
        print(f"  {pair:<28} {ATTACK_LABELS[atk]:<18} "
              f"{res['asr1']*100:6.1f}% {res['n1_attacked']:>6} "
              f"{res['asr2']*100:6.1f}% {res['n2_attacked']:>6} "
              f"{z_str:>7} {p_str:>9} {sig:>6}")
    sep("-")
print(f"\n  Summary: Bonferroni-significant={n_sig_bonf_z}/{total_tests}  "
      f"uncorrected-only={n_sig_uncorr_z}/{total_tests}  "
      f"not-significant={total_tests-n_sig_bonf_z-n_sig_uncorr_z}/{total_tests}")

if mcnemar_available:
    hdr(f"9. McNEMAR'S TEST (paired, TP intersection)  "
        f"Bonferroni α_B={BONFERRONI_ALPHA:.4f}")
    print(f"  {'Model pair':<28} {'Attack':<18} "
          f"{'n_shared':>9} {'b':>5} {'c':>5} {'chi2':>8} {'p':>9} {'sig':>6}")
    sep("-")
    n_sig_bonf_m = 0
    n_sig_uncorr_m = 0
    for atk in ATTACKS:
        if "_status" in mcnemar_results.get(atk, {}):
            print(f"  {ATTACK_LABELS[atk]}: {mcnemar_results[atk].get('_status', 'skipped')}")
            sep("-")
            continue
        for pair, res in mcnemar_results[atk].items():
            n_sh = res.get("intersection_size", 0)
            b    = res.get("b", 0)
            c    = res.get("c", 0)
            chi  = res.get("chi2")
            p    = res.get("p_value")
            if res.get("significant_bonferroni"):
                sig = " * "
                n_sig_bonf_m += 1
            elif res.get("significant_uncorrected"):
                sig = " † "
                n_sig_uncorr_m += 1
            else:
                sig = "   "
            chi_s = f"{chi:.6f}" if chi is not None else "  n/a"
            p_s   = f"{p:.5f}"   if p   is not None else "  n/a  "
            print(f"  {pair:<28} {ATTACK_LABELS[atk]:<18} "
                  f"{n_sh:>9} {b:>5} {c:>5} {chi_s:>8} {p_s:>9} {sig:>6}")
        sep("-")
    print(f"\n  Summary: Bonferroni-significant={n_sig_bonf_m}/{total_tests}  "
          f"uncorrected-only={n_sig_uncorr_m}/{total_tests}")

    hdr("10. AGREEMENT: Z-TEST vs McNEMAR")
    agree = 0
    disagree = 0
    disagree_list = []
    for atk in ATTACKS:
        if "_status" in mcnemar_results.get(atk, {}):
            continue
        for pair in significance_results[atk]:
            z_sig = significance_results[atk][pair].get("significant_bonferroni", False)
            m_sig = mcnemar_results[atk].get(pair, {}).get("significant_bonferroni", False)
            if z_sig == m_sig:
                agree += 1
            else:
                disagree += 1
                disagree_list.append((atk, pair, z_sig, m_sig,
                                      mcnemar_results[atk].get(pair, {}).get("intersection_size", "?")))
    total_c = agree + disagree
    if disagree_list:
        print("  DISAGREEMENTS:")
        for atk, pair, z_sig, m_sig, n_sh in disagree_list:
            print(f"    {ATTACK_SHORT[atk]:<12} {pair}  "
                  f"z-test={'sig' if z_sig else 'ns'}  "
                  f"McNemar={'sig' if m_sig else 'ns'}  "
                  f"(n_shared={n_sh})")
    else:
        print("  All pairs agree between z-test and McNemar.")
    if total_c:
        print(f"\n  Agreement: {agree}/{total_c} ({agree/total_c*100:.0f}%)")
else:
    hdr("9. McNEMAR'S TEST")
    print("  SKIPPED — per_graph data not available.")
    print("  To enable: re-run attack scripts (they save subgraph_indices + evaded).")

hdr("11. FEATURE IMPORTANCE — TOP-10 FEATURES PER MODEL")
print("\n  Gradient-based ranking:")
for m in MODEL_KEYS:
    scores  = np.array(sal_raw["models"][m]["feature_importance"]["gradient_scores"])
    top_idx = np.argsort(scores)[::-1][:10]
    vals    = scores[top_idx]
    feat_str = "  ".join(f"F{i}({v:.6f})" for i, v in zip(top_idx, vals))
    print(f"  {MODEL_LABELS[m]:<12}: {feat_str}")

print("\n  Integrated Gradients ranking:")
for m in MODEL_KEYS:
    scores  = np.array(sal_ig_raw["models"][m]["feature_importance"]["ig_scores"])
    top_idx = np.argsort(scores)[::-1][:10]
    vals    = scores[top_idx]
    feat_str = "  ".join(f"F{i}({v:.6f})" for i, v in zip(top_idx, vals))
    print(f"  {MODEL_LABELS[m]:<12}: {feat_str}")

all_top_grad = [set(np.argsort(
                    np.array(sal_raw["models"][m]["feature_importance"]["gradient_scores"]))[::-1][:10])
                for m in MODEL_KEYS]
consensus_grad = set.intersection(*all_top_grad)
all_top_ig = [set(np.argsort(
                    np.array(sal_ig_raw["models"][m]["feature_importance"]["ig_scores"]))[::-1][:10])
              for m in MODEL_KEYS]
consensus_ig = set.intersection(*all_top_ig)
print(f"\n  Consensus features (top-10 across ALL 4 models):")
print(f"    Gradient: {sorted(consensus_grad) if consensus_grad else 'none'}")
print(f"    IG:       {sorted(consensus_ig)   if consensus_ig   else 'none'}")

print_transfer_adjusted()
print_cross_attack_overlap()

hdr("12. QUICK CONCLUSIONS")
most_robust  = MODEL_LABELS[robustness_rank[0]]
least_robust = MODEL_LABELS[robustness_rank[-1]]
best_atk     = attack_rank[0]
worst_atk    = attack_rank[-1]
print(f"  Most robust model   : {most_robust}  "
      f"(mean ASR {mean_asr[robustness_rank[0]]*100:.1f}%)")
print(f"  Least robust model  : {least_robust}  "
      f"(mean ASR {mean_asr[robustness_rank[-1]]*100:.1f}%)")
print(f"  Most effective atk  : {ATTACK_LABELS[best_atk]}  "
      f"(mean ASR {mean_per_attack[best_atk]*100:.1f}%)")
print(f"  Least effective atk : {ATTACK_LABELS[worst_atk]}  "
      f"(mean ASR {mean_per_attack[worst_atk]*100:.1f}%)")
print(f"  Stat. significant pairs (Bonferroni): {n_sig_bonf_z}/{total_tests}")
print(f"  McNemar available   : {mcnemar_available}")
print(f"  Statistical test methodology:")
print(f"    Primary:      two-proportion z-test (independent proportions)")
print(f"    Supplementary: McNemar chi-squared (paired, TP intersection)")
print(f"    Correction:   Bonferroni ({N_MODEL_PAIRS} pairs per attack)")

sep()
print(f"Outputs saved to : {RESULTS}")
print(f"Plots saved to   : {PLOTS}")
sep()