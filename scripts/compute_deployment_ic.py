"""
Compute per-alpha IC during deployment window.

IC definition consistent with C2/C3 internal computation (alpha_filter.py):
  IC = Spearman correlation between alpha value and next-day log_return.

For each generation event, IC is measured from the day after generation
until the next generation event (i.e., the period the alpha is live).

Usage:
    python scripts/compute_deployment_ic.py
    python scripts/compute_deployment_ic.py --tickers AAPL HSBC
"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.features.alpha_executor import evaluate_alpha_expression
from src.features.technical import compute_all_indicators

warnings.filterwarnings("ignore", category=RuntimeWarning)

TICKERS = ["AAPL", "HSBC", "PEP", "TM", "TCEHY"]


def compute_deployment_ic(ticker: str) -> dict:
    df = pd.read_csv(f"data/price/{ticker}.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df_ind = compute_all_indicators(df.copy())

    # next-day log_return, consistent with C2/C3 in alpha_filter.py
    df_ind["next_day_return"] = (
        np.log(df_ind["close"] / df_ind["close"].shift(1)).shift(-1)
    )

    with open(f"data/alphas/{ticker}.json") as f:
        alpha_log = json.load(f)

    events = sorted(
        [e for e in alpha_log if e.get("alphas")],
        key=lambda x: x["generated_at_idx"],
    )

    rank_ics = {i: [] for i in range(1, 6)}

    for i, ev in enumerate(events):
        deploy_start = ev["generated_at_idx"] + 1
        deploy_end = (
            events[i + 1]["generated_at_idx"] if i + 1 < len(events) else len(df) - 1
        )
        if deploy_end - deploy_start < 5:
            continue

        deploy_slice = df_ind.iloc[deploy_start:deploy_end].copy()
        target = df_ind["next_day_return"].iloc[deploy_start:deploy_end].values

        for alpha in ev["alphas"]:
            rank = alpha.get("rank")
            expr = alpha.get("expression", "")
            if not rank or not expr:
                continue
            try:
                vals = evaluate_alpha_expression(expr, deploy_slice)
                if vals is None:
                    continue
                vals = np.array(vals, dtype=float)
                mask = ~(np.isnan(vals) | np.isnan(target))
                if mask.sum() < 5:
                    continue
                ic, _ = spearmanr(vals[mask], target[mask])
                if not np.isnan(ic):
                    rank_ics[rank].append(ic)
            except Exception:
                pass

    return {
        r: {"mean_ic": round(float(np.mean(v)), 4), "n": len(v)}
        for r, v in rank_ics.items()
        if v
    }


def print_table(results: dict, tickers: list):
    print(f"\nTable: IC of Five Alphas During Deployment (Company-Specific)\n")
    print(f"{'Alpha':<10}", end="")
    for t in tickers:
        print(f"{t:>10}", end="")
    print()
    print("-" * (10 + 10 * len(tickers)))
    for rank in range(1, 6):
        print(f"Alpha {rank:<4}", end="")
        for t in tickers:
            v = results.get(t, {}).get(rank)
            print(f"{v['mean_ic']:>10.4f}" if v else f"{'N/A':>10}", end="")
        print()
    print()
    print("n (number of deployment windows per alpha):")
    print(f"{'Alpha':<10}", end="")
    for t in tickers:
        print(f"{t:>10}", end="")
    print()
    for rank in range(1, 6):
        print(f"Alpha {rank:<4}", end="")
        for t in tickers:
            v = results.get(t, {}).get(rank)
            print(f"{v['n']:>10}" if v else f"{'N/A':>10}", end="")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=TICKERS)
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    args = parser.parse_args()

    results = {}
    for ticker in args.tickers:
        print(f"Computing {ticker}...", end=" ", flush=True)
        results[ticker] = compute_deployment_ic(ticker)
        print("done")

    print_table(results, args.tickers)

    if args.save:
        out_path = "results/aggregated/deployment_ic.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
