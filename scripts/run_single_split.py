"""
Run single 70/30 split for DA-LLM and ablations A1-A5.
Results go to results/{ID}_s/, kept separate from the main results in results/A1..A5/.

Usage (from repo root, venv active):
    python scripts/run_single_split.py --ablation DALLM
    python scripts/run_single_split.py --ablation A1
    python scripts/run_single_split.py --all
"""
import argparse
import copy
import logging
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_REPO_ROOT, '.env'))
except ImportError:
    pass
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import yaml

TICKERS   = ['AAPL', 'HSBC', 'PEP', 'TM', 'TCEHY']
MODELS    = ['transformer_encdec']
SPLIT     = 'single'

CONFIGS = {
    'DALLM': dict(
        alphas_dir='data/alphas',
        overrides={},
        label='DA-LLM full (C1+C2+C3)',
    ),
    'A1': dict(
        alphas_dir='data/alphas/A1',
        overrides={'alpha_regen.use_regime': False, 'use_c3_filter': True},
        label='A1 - C1 OFF (no regime trigger)',
    ),
    'A2': dict(
        alphas_dir='data/alphas/A2',
        overrides={'use_c2_feedback': False, 'use_c3_filter': True},
        label='A2 - C2 OFF (no per-alpha IC feedback)',
    ),
    'A3': dict(
        alphas_dir='data/alphas/A3',
        overrides={'use_c3_filter': False},
        label='A3 - C3 OFF (no generate-and-filter)',
    ),
    'A4': dict(
        alphas_dir='data/alphas/A4',
        overrides={
            'alpha_regen.use_regime': False,
            'alpha_regen.periodic_frequency': 63,
            'alpha_regen.cooldown_days': 100000,
            'use_c3_filter': True,
        },
        label='A4 - Static alpha (LLM, with C3 filter)',
    ),
    'A5': dict(
        alphas_dir='data/alphas/A5_paper',
        overrides={
            'alpha_regen.use_regime': False,
            'alpha_regen.periodic_frequency': 63,
            'alpha_regen.cooldown_days': 100000,
            'use_c3_filter': False,
        },
        label='A5 - Paper static (hardcoded Table 5)',
    ),
    'A12': dict(
        alphas_dir='data/alphas/A12',
        overrides={'use_c2_feedback': False, 'use_c3_filter': False},
        label='A12 - C1 only (regime regen, no C2, no filter)',
    ),
    'A11': dict(
        alphas_dir='data/alphas/A11',
        overrides={'alpha_regen.use_regime': False, 'use_c2_feedback': False, 'use_c3_filter': True},
        label='A11 - C3 only (filter, no C1, no C2)',
    ),
}


def _apply_overrides(config: dict, overrides: dict) -> None:
    for key, val in overrides.items():
        parts = key.split('.')
        d = config
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val


def _clear_checkpoint(alphas_dir: str) -> None:
    """Delete stale checkpoint so old multi-model results don't bleed in."""
    alphas_tag = alphas_dir.replace('/', '_').strip('_')
    ckpt = os.path.join('results/raw', alphas_tag, 'method_m_checkpoint.pkl')
    if os.path.exists(ckpt):
        os.remove(ckpt)
        print(f'  [cleared checkpoint: {ckpt}]')


def run_ablation_single(ablation_id: str, config_path: str = 'config.yaml') -> None:
    cfg = CONFIGS[ablation_id]
    out_dir = f'results/{ablation_id}_s'
    os.makedirs(out_dir, exist_ok=True)

    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    config = copy.deepcopy(base_config)
    config['data']['tickers'] = TICKERS
    config['data']['alphas_dir'] = cfg['alphas_dir']
    _apply_overrides(config, cfg['overrides'])

    # Clear stale checkpoint (may contain results from other model types)
    _clear_checkpoint(cfg['alphas_dir'])

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )
    logger = logging.getLogger(f'single_split.{ablation_id}')
    logger.info(f'=== Single 70/30 split: {ablation_id} — {cfg["label"]} ===')
    logger.info(f'Output dir: {out_dir}')

    from src.walkforward.wf_orchestrator import WalkForwardOrchestrator
    from src.evaluation.reporter import save_all_results

    orch = WalkForwardOrchestrator(config, logger_=logger)
    results = orch.run_all(tickers=TICKERS, model_types=MODELS, split=SPLIT)

    save_all_results(results, output_dir=out_dir)
    logger.info(f'{ablation_id} done — {len(results)} results saved to {out_dir}/')


def main():
    parser = argparse.ArgumentParser(
        description='Run single 70/30 split for DA-LLM and ablations A1-A5.'
    )
    parser.add_argument('--ablation', choices=list(CONFIGS.keys()),
                        help='Which config to run.')
    parser.add_argument('--all', dest='run_all', action='store_true',
                        help='Run all configs sequentially.')
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()

    if not args.run_all and not args.ablation:
        parser.error('Specify --ablation or --all.')

    targets = list(CONFIGS.keys()) if args.run_all else [args.ablation]
    for t in targets:
        print(f'\n{"="*60}')
        print(f'  {t}: {CONFIGS[t]["label"]}')
        print(f'{"="*60}')
        run_ablation_single(t, config_path=args.config)

    print('\nDone. Results in results/{ID}_s/')


if __name__ == '__main__':
    main()
