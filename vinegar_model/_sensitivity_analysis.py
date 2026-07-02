"""
Sensitivity Analysis for Vinegar Model
=======================================
Computes first-order sensitivity indices for 9 UI parameters
on the overall_score using manual OAT (one-at-a-time) method.

Impact direction: whether increasing the parameter increases (+) or decreases (-) overall_score.
"""

import sys
import csv
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vinegar_model import (
    compute_overall_score,
    parameter_ranges,
)
from vinegar_model.flavor_radar import VinegarState
from vinegar_model.data_baseline import USER_DEFAULTS

PARAM_NAMES = [
    "vinegar_age_months",
    "total_acid",
    "non_volatile_acid",
    "reducing_sugar",
    "total_amino_acid",
    "ethyl_acetate",
    "tmp",
    "acetic_acid",
    "ph",
]

OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "sensitivity_results.csv")
OUTPUT_FIG = os.path.join(os.path.dirname(__file__), "figures", "07_sensitivity_ranking.png")

N_SAMPLES = 512

def baseline_state() -> VinegarState:
    return VinegarState(
        vinegar_age_months=USER_DEFAULTS["vinegar_age_months"],
        total_acid=USER_DEFAULTS["total_acid"],
        non_volatile_acid=USER_DEFAULTS["non_volatile_acid"],
        reducing_sugar=USER_DEFAULTS["reducing_sugar"],
        total_amino_acid=USER_DEFAULTS["total_amino_acid"],
        ethyl_acetate=USER_DEFAULTS["ethyl_acetate"],
        tmp=USER_DEFAULTS["tmp"],
        acetic_acid=USER_DEFAULTS["acetic_acid"],
        ph=USER_DEFAULTS["ph"],
        process="固态发酵",
        raw_material="糯米",
        craft_style="传统",
        enable_ph_dimension=True,
    )

def compute_oat_sensitivity(param_names, n_samples=512):
    """
    OAT sensitivity: for each parameter, sample uniformly in [lo, hi]
    while holding others at baseline. Compute correlation with output.

    Returns dict: param -> (S1 index approximation, impact_direction)
    """
    ranges = parameter_ranges()
    base_state = baseline_state()
    base_score = compute_overall_score(base_state)

    results = {}
    all_deltas = []

    for pname in param_names:
        lo, hi = ranges[pname]
        width = hi - lo

        delta_scores = []
        param_values = np.linspace(lo, hi, n_samples)

        for val in param_values:
            state_dict = {
                "vinegar_age_months": USER_DEFAULTS["vinegar_age_months"],
                "total_acid": USER_DEFAULTS["total_acid"],
                "non_volatile_acid": USER_DEFAULTS["non_volatile_acid"],
                "reducing_sugar": USER_DEFAULTS["reducing_sugar"],
                "total_amino_acid": USER_DEFAULTS["total_amino_acid"],
                "ethyl_acetate": USER_DEFAULTS["ethyl_acetate"],
                "tmp": USER_DEFAULTS["tmp"],
                "acetic_acid": USER_DEFAULTS["acetic_acid"],
                "ph": USER_DEFAULTS["ph"],
                "process": "固态发酵",
                "raw_material": "糯米",
                "craft_style": "传统",
                "enable_ph_dimension": True,
            }
            state_dict[pname] = val
            state = VinegarState.from_dict(state_dict)
            score = compute_overall_score(state)
            delta_scores.append(score - base_score)

        delta_scores = np.array(delta_scores)

        impact = "↑+" if np.mean(delta_scores) > 0 else "↓-"

        range_sensitivity = (np.max(delta_scores) - np.min(delta_scores)) / (hi - lo)
        mean_abs_impact = np.mean(np.abs(delta_scores))
        results[pname] = {
            "mean_abs_impact": mean_abs_impact,
            "range_sensitivity": range_sensitivity,
            "impact_direction": impact,
        }
        all_deltas.append((pname, mean_abs_impact))

    sorted_by_impact = sorted(all_deltas, key=lambda x: x[1], reverse=True)
    for rank, (pname, _) in enumerate(sorted_by_impact, 1):
        results[pname]["rank"] = rank

    return results

def main():
    print("Running OAT Sensitivity Analysis (N={})...".format(N_SAMPLES))
    results = compute_oat_sensitivity(PARAM_NAMES, n_samples=N_SAMPLES)

    sorted_results = sorted(results.items(), key=lambda x: x[1]["rank"])

    print("\n" + "=" * 70)
    print("RANKED SENSITIVITY TABLE (OAT First-Order)")
    print("=" * 70)
    print(f"{'Rank':<5} {'Parameter':<22} {'Mean |ΔScore|':<16} {'Impact':<8}")
    print("-" * 70)
    for pname, data in sorted_results:
        print(f"{data['rank']:<5} {pname:<22} {data['mean_abs_impact']:<16.4f} {data['impact_direction']:<8}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Parameter", "MeanAbsImpact", "RangeSensitivity", "ImpactDirection"])
        for pname, data in sorted_results:
            writer.writerow([data["rank"], pname, round(data["mean_abs_impact"], 6),
                           round(data["range_sensitivity"], 6), data["impact_direction"]])

    print("\nCSV saved: {}".format(OUTPUT_CSV))

    try:
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(OUTPUT_FIG), exist_ok=True)

        names = [x[0] for x in sorted_results]
        impacts = [x[1]["mean_abs_impact"] for x in sorted_results]
        colors = ["#d6915a" if x[1]["impact_direction"] == "↑+" else "#5a8fd6"
                  for x in sorted_results]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(names, impacts, color=colors, edgecolor="#333", linewidth=0.5)
        ax.set_xlabel("Mean |Δ Overall Score|", fontsize=11)
        ax.set_title("OAT Sensitivity Ranking: Parameters Affecting Overall Score", fontsize=13, pad=12)
        ax.invert_yaxis()

        for bar, val in zip(bars, impacts):
            ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                   f"{val:.3f}", va="center", fontsize=9)

        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor="#d6915a", label="Increasing → Higher Score"),
            plt.Rectangle((0, 0), 1, 1, facecolor="#5a8fd6", label="Increasing → Lower Score"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

        plt.tight_layout()
        fig.savefig(OUTPUT_FIG, dpi=150, bbox_inches="tight")
        print("Figure saved: {}".format(OUTPUT_FIG))
    except Exception as e:
        print("Warning: Could not generate figure: {}".format(e))

    print("\n" + "=" * 70)
    top_param = sorted_results[0]
    bottom_param = sorted_results[-1]
    print("HIGHEST IMPACT: {} (rank {}, mean|Δ|={:.4f})".format(
        top_param[0], top_param[1]["rank"], top_param[1]["mean_abs_impact"]))
    print("LOWEST IMPACT:  {} (rank {}, mean|Δ|={:.4f})".format(
        bottom_param[0], bottom_param[1]["rank"], bottom_param[1]["mean_abs_impact"]))

    return results

if __name__ == "__main__":
    results = main()