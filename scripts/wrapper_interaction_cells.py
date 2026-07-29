"""Decomposes the replicated wrapper-swap ANOVA's core x category interaction
(results/wrapper_swap_replication.json) down to individual cells: which specific
(core request, wrapper category) pairs actually drive Llama-3.1-8B's significant
interaction term (eta_sq=0.424, p<0.0001), versus additive-model prediction. Pure
re-analysis of already-saved activation grids, no new GPU generation.

Usage: python scripts/wrapper_interaction_cells.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
IN_PATH = RESULTS_DIR / "wrapper_swap_replication.json"
OUT_PATH = RESULTS_DIR / "wrapper_interaction_cells.json"


def cell_interactions(grid: np.ndarray) -> np.ndarray:
    grand_mean = grid.mean()
    row_means = grid.mean(axis=(1, 2))
    col_means = grid.mean(axis=(0, 2))
    cell_means = grid.mean(axis=2)
    fitted = row_means[:, None] + col_means[None, :] - grand_mean
    return cell_means - fitted


def analyze(model_data: dict) -> dict:
    core_requests = model_data["core_requests"]
    category_keys = model_data["category_keys"]
    grid = np.array(model_data["activation_grid"])

    interactions = cell_interactions(grid)
    cell_means = grid.mean(axis=2)

    cells = []
    for i, core in enumerate(core_requests):
        for j, cat in enumerate(category_keys):
            cells.append({
                "core_request": core,
                "category": cat,
                "cell_mean": round(float(cell_means[i, j]), 4),
                "interaction": round(float(interactions[i, j]), 4),
            })
    cells.sort(key=lambda c: abs(c["interaction"]), reverse=True)

    per_core_spread = [
        {
            "core_request": core,
            "interaction_range": round(float(interactions[i].max() - interactions[i].min()), 4),
            "interaction_sd": round(float(interactions[i].std(ddof=1)), 4),
        }
        for i, core in enumerate(core_requests)
    ]
    per_core_spread.sort(key=lambda c: c["interaction_range"], reverse=True)

    return {
        "top_cells": cells[:12],
        "per_core_spread_ranked": per_core_spread,
        "interaction_sd_across_all_cells": round(float(interactions.std(ddof=1)), 4),
    }


def main() -> None:
    with open(IN_PATH, encoding="utf-8") as fh:
        data = json.load(fh)

    out = {model: analyze(model_data) for model, model_data in data.items()}

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    for model, result in out.items():
        print(f"\n=== {model} ===")
        print(f"interaction SD across all 40 cells: {result['interaction_sd_across_all_cells']}")
        print("Top 5 cells by |interaction|:")
        for c in result["top_cells"][:5]:
            print(f"  {c['interaction']:+.3f}  [{c['category']}]  {c['core_request'][:70]}")
        print("Top 3 core requests by interaction range across categories:")
        for c in result["per_core_spread_ranked"][:3]:
            print(f"  range={c['interaction_range']:.3f}  {c['core_request'][:70]}")


if __name__ == "__main__":
    main()
