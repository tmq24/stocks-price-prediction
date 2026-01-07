"""
Stock Close Price Prediction Benchmark Pipeline

5 Models: transformer_encoder, transformer_decoder, vanilla_transformer, lstm, nbeats
Split (date-based):
  - Train: 2015-08-10 to 2021-12-31
  - Val: 2022-01-01 to 2023-12-31
  - Test: 2024-01-01 to 2025-06-20
Test: Evaluated separately per ticker
Models: Saved to models/
Plots: Generated for sample tickers (AAPL, PEP)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ================================================================
# CONFIGURATION
# ================================================================

MODELS = [
    'transformer_encoder',
    'transformer_decoder',
    'vanilla_transformer',
    'lstm',
    'nbeats'
]

INPUT_WINDOWS = [5, 10, 15]
OUTPUT_HORIZONS = [1, 5, 10]

STOCKS = ['AAPL', 'HSBC', 'PEP', 'TM', 'TCEHY']
SAMPLE_TICKERS = ['AAPL', 'PEP']


# ================================================================
# PIPELINE FUNCTIONS
# ================================================================

def run_full_benchmark(verbose: bool = True, generate_plots: bool = True):
    """Run the full benchmark with per-ticker evaluation."""
    from src.benchmark import run_benchmark
    
    print("=" * 60)
    print("STOCK CLOSE PRICE PREDICTION BENCHMARK")
    print("=" * 60)
    print(f"Models: {len(MODELS)} ({', '.join(MODELS)})")
    print(f"Input Windows: {INPUT_WINDOWS}")
    print(f"Output Horizons: {OUTPUT_HORIZONS}")
    print(f"Stocks: {STOCKS}")
    print(f"Plot Samples: {SAMPLE_TICKERS}")
    print(f"Split: Train(~2021) / Val(2022-2023) / Test(2024+)")
    total = len(MODELS) * len(INPUT_WINDOWS) * len(OUTPUT_HORIZONS)
    print(f"Total Experiments: {total}")
    print("=" * 60)
    
    results = run_benchmark(
        models=MODELS,
        input_windows=INPUT_WINDOWS,
        output_horizons=OUTPUT_HORIZONS,
        stocks=STOCKS,
        output_dir="evaluation",
        verbose=verbose,
        generate_plots=generate_plots,
        plot_tickers=SAMPLE_TICKERS
    )
    
    return results


def run_quick_test():
    """Run a quick test with subset of configurations."""
    from src.benchmark import run_benchmark
    
    print("Running quick test...")
    
    results = run_benchmark(
        models=['lstm', 'nbeats'],
        input_windows=[10],
        output_horizons=[1],
        stocks=STOCKS,
        output_dir="evaluation",
        verbose=True,
        generate_plots=True,
        plot_tickers=SAMPLE_TICKERS
    )
    
    return results


def run_single_experiment(model: str, window: int, horizon: int):
    """Run a single experiment with per-ticker evaluation."""
    from src.benchmark import BenchmarkRunner
    
    runner = BenchmarkRunner(
        models=[model],
        input_windows=[window],
        output_horizons=[horizon],
        stocks=STOCKS,
        plot_tickers=SAMPLE_TICKERS
    )
    
    results = runner.run_single_experiment(
        model_type=model,
        window_size=window,
        horizon=horizon,
        verbose=True,
        generate_plots=True
    )
    
    print("\nPer-Ticker Results:")
    for r in results:
        print(f"  {r['ticker']}: MAE={r['mae']:.6f}, MSE={r['mse']:.6f}")
    
    return results


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Stock Close Price Prediction Benchmark'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run full benchmark')
    bench_parser.add_argument('--quiet', action='store_true', 
                              help='Reduce output verbosity')
    bench_parser.add_argument('--no-plots', action='store_true',
                              help='Skip plot generation')
    
    # Quick test command
    subparsers.add_parser('test', help='Run quick test')
    
    # Single experiment command
    single_parser = subparsers.add_parser('single', help='Run single experiment')
    single_parser.add_argument('--model', type=str, required=True,
                               choices=MODELS, help='Model type')
    single_parser.add_argument('--window', type=int, required=True,
                               help='Input window size')
    single_parser.add_argument('--horizon', type=int, required=True,
                               help='Output horizon')
    
    args = parser.parse_args()
    
    if args.command == 'benchmark':
        run_full_benchmark(
            verbose=not args.quiet, 
            generate_plots=not args.no_plots
        )
    
    elif args.command == 'test':
        run_quick_test()
    
    elif args.command == 'single':
        run_single_experiment(
            model=args.model,
            window=args.window,
            horizon=args.horizon
        )
    
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python main.py benchmark              # Run full benchmark")
        print("  python main.py benchmark --no-plots   # Skip plot generation")
        print("  python main.py test                   # Quick test with 2 models")
        print("  python main.py single --model lstm --window 10 --horizon 1")


if __name__ == "__main__":
    main()
