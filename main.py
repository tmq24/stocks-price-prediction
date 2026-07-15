import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(
        description='DA-LLM - LLM-generated dynamic formulaic alphas for stock prediction'
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    thesis_parser = subparsers.add_parser('thesis', help='Run DA-LLM 70/30 experiment')
    thesis_subparsers = thesis_parser.add_subparsers(dest='thesis_command')

    thesis_parser.add_argument('--config', type=str, default='config.yaml')
    thesis_parser.add_argument('--tickers', nargs='+', default=None)
    thesis_parser.add_argument('--models', nargs='+', default=None)
    thesis_parser.add_argument('--ablation', type=str, default=None,
                               choices=['A1', 'A2', 'A3', 'A4', 'A5', 'A8', 'A9', 'A10', 'A11', 'A12'])
    thesis_parser.add_argument('--save-models', action='store_true',
                               help='Save trained state_dict + training history per (ticker, seed). '
                                    'Outputs go to models/<model_type>/<TICKER>/seed_<N>.pth')
    thesis_parser.add_argument('--save-models-dir', type=str, default='models',
                               help='Root directory for --save-models (default: models)')
    thesis_parser.add_argument('--output-dir', type=str, default='results/aggregated',
                               help='Directory for result CSVs (default: results/aggregated)')

    gen_alphas_parser = thesis_subparsers.add_parser(
        'gen-alphas',
        help='Generate and store DA-LLM alpha expressions without running model training',
    )
    gen_alphas_parser.add_argument('--config', type=str, default='config.yaml')
    gen_alphas_parser.add_argument('--tickers', nargs='+', default=None)
    gen_alphas_parser.add_argument('--batch', type=int, default=None,
                                   help='Run tickers in batch N (1-indexed). Overrides --tickers.')
    gen_alphas_parser.add_argument('--batch-size', type=int, default=5,
                                   help='Tickers per batch (default: 5)')

    args = parser.parse_args()

    if args.command == 'thesis':
        thesis_command = getattr(args, 'thesis_command', None)

        if thesis_command == 'gen-alphas':
            _cmd_gen_alphas(args)
        else:
            from src.thesis_runner import run_thesis_experiment, run_ablation
            save_dir = args.save_models_dir if args.save_models else None
            if args.ablation:
                run_ablation(
                    ablation_id=args.ablation,
                    config_path=args.config,
                    tickers=args.tickers,
                    model_types=args.models,
                    split='single',
                )
            else:
                run_thesis_experiment(
                    config_path=args.config,
                    tickers=args.tickers,
                    model_types=args.models,
                    split='single',
                    save_models_dir=save_dir,
                    output_dir=args.output_dir,
                )
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python main.py thesis                                          # 70/30, all 5 tickers")
        print("  python main.py thesis --tickers AAPL                           # AAPL only, 70/30")
        print("  python main.py thesis --ablation A1                           # ablation: drop regime trigger")
        print("  python main.py thesis gen-alphas --tickers AAPL               # alpha gen only")


def _load_config(config_path: str) -> dict:
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _cmd_gen_alphas(args) -> None:
    from src.walkforward.wf_orchestrator import WalkForwardOrchestrator
    config = _load_config(args.config)

    tickers = args.tickers
    if args.batch is not None:
        all_tickers = config['data']['tickers']
        batch_size = args.batch_size
        start = (args.batch - 1) * batch_size
        tickers = all_tickers[start:start + batch_size]
        if not tickers:
            total_batches = -(-len(all_tickers) // batch_size)
            print(f"Batch {args.batch} out of range - only {total_batches} batches total.")
            return
        print(f"Batch {args.batch}/{-(-len(all_tickers) // batch_size)}: {tickers}")

    orchestrator = WalkForwardOrchestrator(config)
    orchestrator.generate_alphas_only(tickers=tickers)


if __name__ == "__main__":
    main()
