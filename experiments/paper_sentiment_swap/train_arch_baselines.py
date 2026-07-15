"""
Train LSTM + N-BEATS architecture baselines on the A2 (C1+C3, polarity) alpha
features — same alphas as the DA-LLM transformer, only the model differs.
Reuses cached A2 alphas (copied to data/alphas/A2_arch) -> NO LLM/API calls.

Output: results/arch_baselines_A2/ (aggregated_results.csv + predictions.pkl)
"""
import copy, os, pickle, sys
from collections import defaultdict

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
try:
    from dotenv import load_dotenv; load_dotenv(os.path.join(_REPO, '.env'))
except ImportError:
    pass
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
import logging, yaml
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

TICKERS = ['AAPL', 'HSBC', 'PEP', 'TM', 'TCEHY']
OUT = os.path.join(_REPO, 'results', 'arch_baselines_A2')


def main():
    cfg = copy.deepcopy(yaml.safe_load(open(os.path.join(_REPO, 'config.yaml'))))
    cfg['data']['alphas_dir'] = 'data/alphas/A2_arch'   # reuse A2 alphas (cached)
    cfg['data']['tickers'] = TICKERS
    cfg['use_c2_feedback'] = False                       # A2 = C2 off (alphas cached anyway)
    cfg['models'] = {k: v for k, v in cfg['models'].items() if k in ('lstm', 'nbeats')}
    os.makedirs(OUT, exist_ok=True)

    from src.walkforward.wf_orchestrator import WalkForwardOrchestrator
    orch = WalkForwardOrchestrator(cfg, logger_=logging.getLogger('arch'))
    results = orch.run_all(tickers=TICKERS, split='single')

    from src.evaluation.reporter import build_results_dataframe, aggregate_by_configuration
    flat = build_results_dataframe(results)
    flat.to_csv(os.path.join(OUT, 'fold_results.csv'), index=False)
    aggregate_by_configuration(flat).to_csv(os.path.join(OUT, 'aggregated_results.csv'), index=False)
    preds = defaultdict(list)
    for r in results:
        preds[(r.fold_idx, r.ticker, r.model_type)].append(r)
    with open(os.path.join(OUT, 'predictions.pkl'), 'wb') as f:
        pickle.dump(dict(preds), f)
    print(f"written -> {OUT}")


if __name__ == '__main__':
    main()
