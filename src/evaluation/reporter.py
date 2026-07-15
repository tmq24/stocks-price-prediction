"""
Aggregate FoldResult objects into summary tables and export to CSV / LaTeX.
"""
import os
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import pandas as pd


def build_results_dataframe(fold_results: List) -> pd.DataFrame:
    """Convert a list of FoldResult objects to a flat DataFrame."""
    rows = []
    for r in fold_results:
        rows.append({
            'fold_idx': r.fold_idx,
            'ticker': r.ticker,
            'model_type': r.model_type,
            'horizon': r.horizon,
            'seed': r.seed,
            'mse': r.mse,
            'mse_scaled': r.mse_scaled,
            'rmse': r.rmse,
            'mae': r.mae,
            'directional_accuracy': r.directional_accuracy,
            'pearson_ic': r.pearson_ic,
            'spearman_rank_ic': r.spearman_rank_ic,
            'n_train_sequences': r.n_train_sequences,
            'n_test_points': len(r.test_preds),
        })
    return pd.DataFrame(rows)


def aggregate_by_configuration(
    df: pd.DataFrame,
    group_by: List[str] = None,
) -> pd.DataFrame:
    """
    Group by configuration and compute mean ± std over folds and seeds.

    Returns a wide-format table with columns like:
        mse_mean, mse_std, pearson_ic_mean, pearson_ic_std, ...
    """
    if group_by is None:
        group_by = ['ticker', 'model_type', 'horizon']

    metric_cols = ['mse', 'mse_scaled', 'rmse', 'mae', 'directional_accuracy', 'pearson_ic', 'spearman_rank_ic']

    agg_dict = {}
    for col in metric_cols:
        agg_dict[f'{col}_mean'] = (col, 'mean')
        agg_dict[f'{col}_std'] = (col, 'std')

    grouped = df.groupby(group_by)[metric_cols].agg(['mean', 'std'])
    grouped.columns = [f'{col}_{stat}' for col, stat in grouped.columns]
    return grouped.reset_index()


def format_mean_std(df: pd.DataFrame, metric: str, decimals: int = 4) -> pd.Series:
    """Return a formatted 'mean ± std' Series for a metric column."""
    mean_col = f'{metric}_mean'
    std_col = f'{metric}_std'
    fmt = f'{{:.{decimals}f}}'
    return df.apply(
        lambda row: f"{fmt.format(row[mean_col])} ± {fmt.format(row[std_col])}",
        axis=1,
    )


def export_latex_table(
    agg_df: pd.DataFrame,
    filepath: str,
    metric: str = 'pearson_ic',
    caption: str = '',
    label: str = '',
) -> None:
    """Export an aggregated results table as a LaTeX table."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    display = agg_df.copy()
    display['IC (mean ± std)'] = format_mean_std(display, 'pearson_ic')
    display['DA (mean ± std)'] = format_mean_std(display, 'directional_accuracy')
    display['MSE (mean ± std)'] = format_mean_std(display, 'mse')

    cols_to_show = ['ticker', 'model_type', 'horizon',
                    'IC (mean ± std)', 'DA (mean ± std)', 'MSE (mean ± std)']
    cols_to_show = [c for c in cols_to_show if c in display.columns]

    latex = display[cols_to_show].to_latex(
        index=False,
        escape=True,
        caption=caption or f"Results aggregated by configuration ({metric})",
        label=label or f"tab:{metric}_results",
    )
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(latex)


def generate_ablation_report(
    all_results: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Combine multiple ablation result DataFrames into a single comparison table.

    Args:
        all_results: dict mapping ablation label -> aggregated DataFrame

    Returns a wide DataFrame with ablation label as an additional column.
    """
    frames = []
    for label, df in all_results.items():
        df = df.copy()
        df['ablation'] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save_all_results(
    fold_results: List,
    output_dir: str = 'results/aggregated',
) -> None:
    """Convenience: save flat CSV and aggregated CSV to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    flat_df = build_results_dataframe(fold_results)
    flat_df.to_csv(os.path.join(output_dir, 'fold_results.csv'), index=False)

    agg_df = aggregate_by_configuration(flat_df)
    agg_df.to_csv(os.path.join(output_dir, 'aggregated_results.csv'), index=False)

    export_latex_table(
        agg_df,
        filepath=os.path.join(output_dir, 'main_table.tex'),
    )
