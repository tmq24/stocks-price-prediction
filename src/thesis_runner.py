"""
Top-level entry point for the DA-LLM thesis experiment.

Usage:
    python main.py thesis                          # full run
    python main.py thesis --ablation A1            # specific ablation
    python main.py thesis --tickers AAPL MSFT      # subset of tickers
    python main.py thesis --config my_config.yaml  # custom config
"""
import logging
import os
import random
import sys
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str = 'config.yaml') -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: str = 'logs') -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'thesis_{timestamp}.log')

    logger = logging.getLogger('thesis')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logging to {log_file}")
    return logger


def set_global_seeds(seeds: List[int]) -> None:
    """Set all RNG seeds for reproducibility.  Called once per top-level run."""
    for seed in seeds[:1]:  # use first seed for global state
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_thesis_experiment(
    config_path: str = 'config.yaml',
    tickers: Optional[List[str]] = None,
    model_types: Optional[List[str]] = None,
    split: str = 'single',
    save_models_dir: Optional[str] = None,
    output_dir: str = 'results/aggregated',
) -> None:
    from .walkforward.wf_orchestrator import WalkForwardOrchestrator
    from .evaluation.reporter import save_all_results

    config = load_config(config_path)
    logger = setup_logging()

    # config can override output_dir; CLI arg takes precedence if explicitly set
    output_dir = output_dir or config.get('output_dir', 'results/aggregated')

    seeds = config.get('training', {}).get('seeds', [42])
    set_global_seeds(seeds)

    logger.info('-' * 40)
    logger.info("DA-LLM: LLM-Generated Dynamic Formulaic Alphas")
    logger.info('-' * 40)
    if save_models_dir:
        logger.info(f"Model checkpoints will be saved under: {save_models_dir}")

    orchestrator = WalkForwardOrchestrator(config, logger_=logger)
    fold_results = orchestrator.run_all(
        tickers=tickers,
        model_types=model_types,
        split=split,
        save_models_dir=save_models_dir,
    )

    logger.info(f"Total fold results collected: {len(fold_results)}")

    save_all_results(fold_results, output_dir=output_dir)
    logger.info(f"Results saved to {output_dir}/")
    logger.info("DA-LLM experiment complete.")


# ---------------------------------------------------------------------------
# Ablation runners
# ---------------------------------------------------------------------------

def run_ablation(
    ablation_id: str,
    config_path: str = 'config.yaml',
    tickers: Optional[List[str]] = None,
    model_types: Optional[List[str]] = None,
    split: str = 'single',
) -> None:
    """Run a specific ablation study (A1=C1 OFF, A2=C2 OFF, A3=C3 OFF, A4=STATIC)."""
    from .walkforward.wf_orchestrator import WalkForwardOrchestrator
    from .evaluation.reporter import save_all_results

    config = load_config(config_path)
    logger = setup_logging()
    set_global_seeds(config.get('training', {}).get('seeds', [42]))

    logger.info(f"Running ablation {ablation_id}")

    import copy
    config_copy = copy.deepcopy(config)

    if ablation_id == 'A1':
        # C1 OFF: disable regime-conditional regen, use periodic-only trigger.
        config_copy['alpha_regen']['use_regime'] = False
        config_copy['use_c3_filter'] = True
        config_copy['data']['alphas_dir'] = 'data/alphas/A1'
        logger.info("A1: C1 OFF (no regime trigger) -> results/A1/")

    elif ablation_id == 'A2':
        # C2 OFF: no per-alpha IC feedback in LLM prompt.
        config_copy['use_c2_feedback'] = False
        config_copy['use_c3_filter'] = True
        config_copy['data']['alphas_dir'] = 'data/alphas/A2'
        logger.info("A2: C2 OFF (aggregate IC only) -> results/A2/")

    elif ablation_id == 'A3':
        # C3 OFF: generate 5 directly, no IC-based top-K filter.
        config_copy['use_c3_filter'] = False
        config_copy['data']['alphas_dir'] = 'data/alphas/A3'
        logger.info("A3: C3 OFF (no generate-and-filter) -> results/A3/")

    elif ablation_id == 'A4':
        # A4 = STATIC alpha (paper-like): generate ONCE at idx=63, never regen.
        # Disables C1 (regime) + makes cooldown effectively infinite.
        config_copy['alpha_regen']['use_regime'] = False
        config_copy['alpha_regen']['periodic_frequency'] = 63   # first gen only
        config_copy['alpha_regen']['cooldown_days'] = 100000    # block all subsequent regen
        config_copy['use_c3_filter'] = True
        config_copy['data']['alphas_dir'] = 'data/alphas/A4'
        logger.info("A4: STATIC (single alpha set, no regen) -> results/A4/")

    elif ablation_id == 'A5':
        # A5 = PAPER STATIC: hardcoded Table 5 formulas (Chen & Kawashima 2025),
        # loaded from data/alphas/A5_paper/<TICKER>.json. Same freeze logic as A4.
        # Orchestrator fast-path reuses the cached expressions -> no LLM call.
        config_copy['alpha_regen']['use_regime'] = False
        config_copy['alpha_regen']['periodic_frequency'] = 63
        config_copy['alpha_regen']['cooldown_days'] = 100000
        config_copy['use_c3_filter'] = False  # no candidate pool to filter
        config_copy['data']['alphas_dir'] = 'data/alphas/A5_paper'
        logger.info("A5: PAPER STATIC (Table 5 hardcoded) -> results/A5/")

    elif ablation_id == 'A8':
        # A8 = VANILLA DYNAMIC: all 3 contributions OFF.
        # Periodic-only trigger, simple prompt, no candidate-pool filter.
        # Tests whether dynamic regen alone (without C1+C2+C3) is enough.
        config_copy['alpha_regen']['use_regime'] = False
        config_copy['use_c2_feedback'] = False
        config_copy['use_c3_filter'] = False
        config_copy['data']['alphas_dir'] = 'data/alphas/A8'
        logger.info("A8: VANILLA DYNAMIC (all 3 OFF, periodic regen) -> results/A8/")

    elif ablation_id == 'A9':
        # A9 = STATIC WITHOUT FILTER: single LLM gen at idx=63, no C3 filter.
        # Control for A4 (static WITH filter) to isolate filter contribution in static mode.
        config_copy['alpha_regen']['use_regime'] = False
        config_copy['alpha_regen']['periodic_frequency'] = 63
        config_copy['alpha_regen']['cooldown_days'] = 100000
        config_copy['use_c3_filter'] = False
        config_copy['data']['alphas_dir'] = 'data/alphas/A9'
        logger.info("A9: STATIC NO FILTER (single gen, no C3) -> results/A9/")

    elif ablation_id == 'A10':
        # A10 = (C1=OFF, C2=ON, C3=OFF) - C2 alone in C1-OFF regime.
        # Paired with A8 (all OFF) isolates C2 effect with C3 controlled OFF;
        # paired with A1 isolates C3 effect with C2 controlled ON.
        config_copy['alpha_regen']['use_regime'] = False
        config_copy['use_c3_filter'] = False
        config_copy['data']['alphas_dir'] = 'data/alphas/A10'
        logger.info("A10: C2 only (no C1, no C3) -> results/A10/")

    elif ablation_id == 'A11':
        # A11 = (C1=OFF, C2=OFF, C3=ON) - C3 alone in C1-OFF regime.
        # Paired with A8 (all OFF) isolates C3 effect with C2 controlled OFF;
        # paired with A1 isolates C2 effect with C3 controlled ON.
        config_copy['alpha_regen']['use_regime'] = False
        config_copy['use_c2_feedback'] = False
        config_copy['data']['alphas_dir'] = 'data/alphas/A11'
        logger.info("A11: C3 only (no C1, no C2) -> results/A11/")

    elif ablation_id == 'A12':
        # A12 = (C1=ON, C2=OFF, C3=OFF) - C1 alone.
        # Paired with A8 (all OFF) isolates C1 effect with C2+C3 controlled OFF.
        config_copy['use_c2_feedback'] = False
        config_copy['use_c3_filter'] = False
        config_copy['data']['alphas_dir'] = 'data/alphas/A12'
        logger.info("A12: C1 only (regime trigger, no C2, no C3) -> results/A12/")

    else:
        logger.error(f"Unknown ablation id: {ablation_id} (valid: A1, A2, A3, A4, A5, A8, A9, A10, A11, A12)")
        return

    orch = WalkForwardOrchestrator(config_copy, logger_=logger)
    results = orch.run_all(tickers=tickers, model_types=model_types, split=split)
    save_all_results(results, output_dir=f'results/{ablation_id}')
    logger.info(f"{ablation_id} done: {len(results)} results")
