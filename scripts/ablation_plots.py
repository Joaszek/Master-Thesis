import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


RESULTS_DIR = "data/results"
OUT_DIR     = os.path.join(RESULTS_DIR, "plots", "ablation")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS      = ["gatv2", "sage", "sage_edge", "gin"]
MODEL_NAMES = {"gatv2": "GATv2", "sage": "SAGE", "sage_edge": "SAGE+Edge", "gin": "GIN"}
COLORS      = {"gatv2": "#e05c5c", "sage": "#f0a830", "sage_edge": "#4a9eda", "gin": "#5bbf72"}
MARKERS     = {"gatv2": "o", "sage": "s", "sage_edge": "^", "gin": "D"}

DPI = 150


def load(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)

def style_ax(ax, ylabel="ASR (%)", xlabel=None, title=None, ylim=(0, 100)):
    ax.set_ylabel(ylabel, fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12, pad=8)
    if ylim:
        ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_pgd():
    data     = load("pgd_attack_results.json")
    epsilons = data["epsilons"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("PGD Feature Attack — Ablation over ε", fontsize=13, y=1.01)

    for m in MODELS:
        er   = data["models"][m]["epsilon_results"]
        asrs = [er[str(e)]["asr"] * 100 for e in epsilons]
        f1s  = [(er[str(e)]["post_f1_macro"] - er[str(e)]["clean_f1_macro"]) * 100
                for e in epsilons]
        kw = dict(color=COLORS[m], marker=MARKERS[m], linewidth=1.8,
                  markersize=6, label=MODEL_NAMES[m])
        axes[0].plot(epsilons, asrs, **kw)
        axes[1].plot(epsilons, f1s, **kw)

    style_ax(axes[0], ylabel="ASR (%)", xlabel="ε (perturbation budget)", title="Attack Success Rate")
    style_ax(axes[1], ylabel="F1-macro drop (pp)", xlabel="ε (perturbation budget)",
             title="F1-macro Drop", ylim=(-40, 2))
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.0f}pp"))
    axes[0].legend(framealpha=0.85, fontsize=9)
    fig.tight_layout()
    save(fig, "pgd_ablation.png")


def plot_saliency():
    data     = load("saliency_attack_results.json")
    top_ks   = data["top_k_values"]
    epsilons = data["epsilons"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    fig.suptitle("Saliency-Guided Attack — Ablation over top_k", fontsize=13, y=1.01)

    for ax, eps in zip(axes, epsilons):
        for m in MODELS:
            asrs = []
            for tk in top_ks:
                er = (data["models"][m]["topk_results"]
                      .get(str(tk), {})
                      .get("epsilon_results", {})
                      .get(str(eps), {}))
                asrs.append((er.get("asr") or 0) * 100)
            ax.plot(top_ks, asrs, color=COLORS[m], marker=MARKERS[m],
                    linewidth=1.8, markersize=6, label=MODEL_NAMES[m])
        style_ax(ax, ylabel="ASR (%)", xlabel="top_k (features perturbed)",
                 title=f"ε = {eps}")
        ax.set_xticks(top_ks)

    axes[0].legend(framealpha=0.85, fontsize=9)
    for ax in axes:
        ax.axvline(x=43, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax.text(43.5, 2, "all feats\n(= PGD)", fontsize=7.5, color="gray")

    fig.tight_layout()
    save(fig, "saliency_ablation.png")


def plot_zoo():
    data     = load("zoo_attack_results.json")
    epsilons = data["epsilons"]
    qbudgets = data["query_budgets"]
    mid_eps  = epsilons[len(epsilons) // 2]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("ZOO Black-Box Attack — Ablation", fontsize=13, y=1.01)

    for m in MODELS:
        asrs = [data["models"][m]["epsilon_results"][str(e)]["asr"] * 100
                for e in epsilons]
        axes[0].plot(epsilons, asrs, color=COLORS[m], marker=MARKERS[m],
                     linewidth=1.8, markersize=6, label=MODEL_NAMES[m])
    style_ax(axes[0], ylabel="ASR (%)", xlabel="ε (perturbation budget)",
             title="ASR vs. ε  (budget = 2 000 queries)")

    for m in MODELS:
        er   = data["models"][m]["epsilon_results"][str(mid_eps)]
        asrs = [er["asr_at_queries"][str(b)] * 100 for b in qbudgets]
        axes[1].plot(qbudgets, asrs, color=COLORS[m], marker=MARKERS[m],
                     linewidth=1.8, markersize=6, label=MODEL_NAMES[m])
    style_ax(axes[1], ylabel="ASR (%)", xlabel="Query budget",
             title=f"ASR vs. Query Budget  (ε = {mid_eps})")
    axes[1].set_xscale("log")
    axes[1].xaxis.set_major_formatter(mticker.ScalarFormatter())
    axes[1].set_xticks(qbudgets)

    axes[0].legend(framealpha=0.85, fontsize=9)
    axes[1].legend(framealpha=0.85, fontsize=9)
    fig.tight_layout()
    save(fig, "zoo_ablation.png")


def plot_injection():
    data       = load("node_injection_results.json")
    k_values   = data["k_nodes"]
    c_values   = data["connections_per_node"]
    strategies = data["strategies"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    fig.suptitle("Node Injection Attack — Ablation over k_nodes x connections_per_node",
                 fontsize=13, y=1.01)

    linestyles = {1: "-", 2: "--", 3: ":"}

    for ax, strategy in zip(axes, strategies):
        for c in c_values:
            asrs = []
            for k in k_values:
                key  = f"{strategy}_k{k}_c{c}"
                vals = [data["models"][m]["combinations"].get(key, {}).get("asr") or 0
                        for m in MODELS]
                asrs.append(np.mean(vals) * 100)
            ax.plot(k_values, asrs, color="#555", linestyle=linestyles[c],
                    marker="o", markersize=5, linewidth=1.6,
                    label=f"conn={c}")
        style_ax(ax, ylabel="avg ASR (%) over models", xlabel="k_nodes",
                 title=strategy.capitalize())
        ax.set_xticks(k_values)
        ax.legend(framealpha=0.85, fontsize=9)

    fig.tight_layout()
    save(fig, "injection_ablation_strategies.png")

    fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    fig2.suptitle("Node Injection (random) — per model", fontsize=13, y=1.01)

    for ax, m in zip(axes2, MODELS):
        for c in c_values:
            asrs = []
            for k in k_values:
                key = f"random_k{k}_c{c}"
                er  = data["models"][m]["combinations"].get(key, {})
                asrs.append((er.get("asr") or 0) * 100)
            ax.plot(k_values, asrs, color=COLORS[m], linestyle=linestyles[c],
                    marker=MARKERS[m], markersize=5, linewidth=1.6,
                    label=f"conn={c}")
        style_ax(ax, ylabel="ASR (%)", xlabel="k_nodes",
                 title=MODEL_NAMES[m])
        ax.set_xticks(k_values)
        ax.legend(framealpha=0.85, fontsize=8)

    fig2.tight_layout()
    save(fig2, "injection_ablation_per_model.png")



def plot_topology():
    data       = load("topology_rewiring_results.json")
    techniques = data["techniques"]
    k_vals     = data["k_intermediates"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Topology Rewiring Attack — Ablation over technique x k", fontsize=13, y=1.01)

    tech_colors = {"peel_chain": "#e05c5c", "star": "#4a9eda", "parallel": "#5bbf72"}
    tech_ls     = {"peel_chain": "-", "star": "--", "parallel": "-."}

    ax = axes[0]
    for tech in techniques:
        asrs = []
        for k in k_vals:
            key  = f"{tech}_k{k}"
            vals = [data["models"][m]["combinations"].get(key, {}).get("asr") or 0
                    for m in MODELS]
            asrs.append(np.mean(vals) * 100)
        ax.plot(k_vals, asrs, color=tech_colors[tech], linestyle=tech_ls[tech],
                marker="o", markersize=6, linewidth=2, label=tech.replace("_", " "))
    style_ax(ax, ylabel="avg ASR (%) over models", xlabel="k_intermediates",
             title="Average over models")
    ax.set_xticks(k_vals)
    ax.legend(framealpha=0.85, fontsize=9)

    ax2     = axes[1]
    k_show  = max(k_vals)
    x       = np.arange(len(MODELS))
    width   = 0.25
    for i, tech in enumerate(techniques):
        key  = f"{tech}_k{k_show}"
        asrs = [(data["models"][m]["combinations"].get(key, {}).get("asr") or 0) * 100
                for m in MODELS]
        ax2.bar(x + i * width, asrs, width, label=tech.replace("_", " "),
                color=tech_colors[tech], alpha=0.85, edgecolor="white")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([MODEL_NAMES[m] for m in MODELS], fontsize=10)
    style_ax(ax2, ylabel="ASR (%)", xlabel=None,
             title=f"Per-model comparison  (k = {k_show})")
    ax2.legend(framealpha=0.85, fontsize=9)

    fig.tight_layout()
    save(fig, "topology_ablation.png")


def plot_transfer():
    import matplotlib.colors as mcolors

    data     = load("transfer_attack_results.json")
    matrix   = data.get("cross_arch_matrix", {})
    epsilons = data["epsilons"]

    surrogates = list(matrix.keys())
    targets = []
    for s in surrogates:
        for eps in epsilons:
            for t in matrix[s].get(str(eps), {}):
                if t not in targets:
                    targets.append(t)

    fig, axes = plt.subplots(1, len(epsilons), figsize=(5 * len(epsilons), 4.5))
    fig.suptitle("Transfer Attack — Cross-Architecture ASR Heatmap", fontsize=13, y=1.01)

    if len(epsilons) == 1:
        axes = [axes]

    for ax, eps in zip(axes, epsilons):
        grid = np.full((len(surrogates), len(targets)), np.nan)
        for i, s in enumerate(surrogates):
            for j, t in enumerate(targets):
                val = matrix[s].get(str(eps), {}).get(t, {})
                asr = val.get("transfer_asr") if isinstance(val, dict) else None
                if asr is not None:
                    grid[i, j] = asr * 100

        im = ax.imshow(grid, vmin=0, vmax=30, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(targets)))
        ax.set_yticks(range(len(surrogates)))
        ax.set_xticklabels([MODEL_NAMES.get(t, t) for t in targets], fontsize=9)
        ax.set_yticklabels([MODEL_NAMES.get(s, s) for s in surrogates], fontsize=9)
        ax.set_xlabel("Target model", fontsize=10)
        if eps == epsilons[0]:
            ax.set_ylabel("Surrogate model", fontsize=10)
        ax.set_title(f"ε = {eps}", fontsize=11)

        for i in range(len(surrogates)):
            for j in range(len(targets)):
                val = grid[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                            fontsize=8.5, color="black" if val < 20 else "white")
                else:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#aaa")

        plt.colorbar(im, ax=ax, label="ASR (%)", shrink=0.85)

    fig.tight_layout()
    save(fig, "transfer_ablation.png")


def main():
    print("Generating ablation plots")
    plot_pgd()
    plot_saliency()
    plot_zoo()
    plot_injection()
    plot_topology()
    plot_transfer()
    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
