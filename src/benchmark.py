import os
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict

from .utils.data_utils import prepare_benchmark_data
from .utils.train import train_and_evaluate_per_ticker


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

BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 5e-4
PATIENCE = 20

OUTPUT_DIR = "evaluation"
PLOT_DIR = "plot"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# plot
def plot_predictions(ticker: str, predictions: np.ndarray, targets: np.ndarray,
                     model_type: str, window_size: int, horizon: int, 
                     dates: np.ndarray = None) -> str:
    """Plot predicted vs actual close price for a ticker."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    n_samples = len(predictions)
    
    if dates is not None and len(dates) == n_samples:
        x = pd.to_datetime(dates)
    else:
        x = np.arange(n_samples)
    
    ax.plot(x, targets, label='Actual', linewidth=1.5, alpha=0.8, color='blue')
    ax.plot(x, predictions, label='Predicted', linewidth=1.5, alpha=0.8, color='orange')
    
    if dates is not None:
        ax.set_xlabel('Date')
        fig.autofmt_xdate()
    else:
        ax.set_xlabel('Sample')
        
    ax.set_ylabel('Close Price ($)')
    ax.set_title(f'{model_type} | {ticker} | Window={window_size}d, Horizon={horizon}d')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_name = f"pred_{model_type}_{ticker}_w{window_size}_h{horizon}.png"
    save_path = os.path.join(PLOT_DIR, save_name)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


# benchmark runner
class BenchmarkRunner:
    """Runs benchmark experiments with per-ticker test evaluation."""
    
    def __init__(self,
                 models: List[str] = None,
                 input_windows: List[int] = None,
                 output_horizons: List[int] = None,
                 stocks: List[str] = None,
                 output_dir: str = "evaluation",
                 plot_tickers: List[str] = None):
        """
        Args:
            plot_tickers: Tickers to generate plots for (default: ['AAPL', 'PEP'])
        """
        self.models = models or MODELS
        self.input_windows = input_windows or INPUT_WINDOWS
        self.output_horizons = output_horizons or OUTPUT_HORIZONS
        self.stocks = stocks or STOCKS
        self.output_dir = output_dir
        self.plot_tickers = plot_tickers or ['AAPL', 'PEP']
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.results = []
        self.data_cache = {}
    
    def _get_cache_key(self, window: int, horizon: int) -> str:
        return f"w{window}_h{horizon}"
    
    def _load_data(self, window: int, horizon: int) -> Dict:
        """Load data with caching."""
        cache_key = self._get_cache_key(window, horizon)
        
        if cache_key not in self.data_cache:
            self.data_cache[cache_key] = prepare_benchmark_data(
                tickers=self.stocks,
                window_size=window,
                horizon=horizon
            )
        
        return self.data_cache[cache_key]
    
    def run_single_experiment(self,
                              model_type: str,
                              window_size: int,
                              horizon: int,
                              verbose: bool = True,
                              generate_plots: bool = True) -> Dict:
        """
        Run a single experiment.
        - Train on combined train/val
        - Test per ticker
        - Save best model
        - Generate plots for sample tickers
        """
        # Load data
        data = self._load_data(window_size, horizon)
        
        start_time = time.time()
        
        # Train and evaluate per ticker
        result = train_and_evaluate_per_ticker(
            model_type=model_type,
            X_train=data['X_train'],
            y_train=data['y_train'],
            X_val=data['X_val'],
            y_val=data['y_val'],
            test_per_ticker=data['test_per_ticker'],
            window_size=window_size,
            horizon=horizon,
            batch_size=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            lr=LEARNING_RATE,
            patience=PATIENCE,
            verbose=verbose,
            save_best=True,
            target_scaler=data.get('target_scaler')
        )
        
        elapsed_time = time.time() - start_time
        
        # Generate plots for sample tickers
        if generate_plots:
            for ticker in self.plot_tickers:
                if ticker in result['results_per_ticker']:
                    ticker_result = result['results_per_ticker'][ticker]
                    
                    # Get dates from test_per_ticker
                    dates = None
                    if ticker in data['test_per_ticker']:
                         ticker_data = data['test_per_ticker'][ticker]
                         if len(ticker_data) >= 3:
                             dates = ticker_data[2]

                    plot_path = plot_predictions(
                        ticker=ticker,
                        predictions=ticker_result['predictions'],
                        targets=ticker_result['targets'],
                        model_type=model_type,
                        window_size=window_size,
                        horizon=horizon,
                        dates=dates
                    )
                    print(f"  Plot saved: {plot_path}")
        
        # Collect results per ticker
        experiment_results = []
        for ticker, ticker_result in result['results_per_ticker'].items():
            experiment_results.append({
                'input_window': window_size,
                'output_horizon': horizon,
                'model': model_type,
                'ticker': ticker,
                'mae': ticker_result['mae'],
                'mse': ticker_result['mse'],
                'corr': ticker_result.get('corr', 0.0),
                'model_path': result['model_path'],
                'elapsed_time': elapsed_time
            })
        
        return experiment_results
    
    def run_all_experiments(self, verbose: bool = True, generate_plots: bool = True) -> pd.DataFrame:
        """Run all benchmark experiments."""
        total_experiments = (
            len(self.models) * 
            len(self.input_windows) * 
            len(self.output_horizons)
        )
        
        print(f"Running {total_experiments} experiments...")
        print(f"Models: {self.models}")
        print(f"Input Windows: {self.input_windows}")
        print(f"Output Horizons: {self.output_horizons}")
        print(f"Stocks: {self.stocks}")
        print(f"Plot Tickers: {self.plot_tickers}")
        print("=" * 60)
        
        experiment_count = 0
        start_total = time.time()
        
        for window in self.input_windows:
            for horizon in self.output_horizons:
                for model_type in self.models:
                    experiment_count += 1
                    
                    print(f"\n[{experiment_count}/{total_experiments}] "
                          f"Model: {model_type}, Window: {window}, "
                          f"Horizon: {horizon}")
                    
                    try:
                        exp_results = self.run_single_experiment(
                            model_type=model_type,
                            window_size=window,
                            horizon=horizon,
                            verbose=verbose,
                            generate_plots=generate_plots
                        )
                        self.results.extend(exp_results)
                    
                    except Exception as e:
                        print(f"  ERROR: {str(e)}")
                        for ticker in self.stocks:
                            self.results.append({
                                'input_window': window,
                                'output_horizon': horizon,
                                'model': model_type,
                                'ticker': ticker,
                                'mae': np.nan,
                                'mse': np.nan,
                                'error': str(e)
                            })
        
        total_time = time.time() - start_total
        print(f"\nTotal time: {total_time/60:.2f} minutes")
        
        results_df = pd.DataFrame(self.results)
        return results_df
    
    def save_results(self, results_df: pd.DataFrame, filename: str = None):
        """Save results to CSV."""
        if filename is None:
            filename = "benchmark_results.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        results_df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
        
        return filepath
    
    def print_summary(self, results_df: pd.DataFrame):
        """Print summary of results."""
        print(f"\n{'='*80}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*80}")
        
        # Average across tickers
        avg_df = results_df.groupby(['input_window', 'output_horizon', 'model']).agg({
            'mae': 'mean',
            'mse': 'mean',
            'corr': 'mean'
        }).reset_index()
        
        print(f"{'Window':<10}{'Horizon':<10}{'Model':<25}{'MAE':<12}{'MSE':<12}{'Corr':<10}")
        print("-" * 80)
        
        for _, row in avg_df.iterrows():
            corr = row.get('corr', 0)
            print(f"{row['input_window']:<10}{row['output_horizon']:<10}"
                  f"{row['model']:<25}{row['mae']:<12.6f}{row['mse']:<12.6f}{corr:<10.3f}")


def run_benchmark(models: List[str] = None,
                  input_windows: List[int] = None,
                  output_horizons: List[int] = None,
                  stocks: List[str] = None,
                  output_dir: str = "evaluation",
                  verbose: bool = True,
                  generate_plots: bool = True,
                  plot_tickers: List[str] = None) -> pd.DataFrame:
    """
    Run the full benchmark.
    
    Args:
        plot_tickers: Tickers to generate plots for (default: ['AAPL', 'PEP'])
    """
    runner = BenchmarkRunner(
        models=models,
        input_windows=input_windows,
        output_horizons=output_horizons,
        stocks=stocks,
        output_dir=output_dir,
        plot_tickers=plot_tickers or ['AAPL', 'PEP']
    )
    
    results_df = runner.run_all_experiments(verbose=verbose, generate_plots=generate_plots)
    
    runner.save_results(results_df, "benchmark_results.csv")
    save_results_markdown(results_df, output_dir)
    runner.print_summary(results_df)
    
    idx = results_df.groupby(['input_window', 'output_horizon'])['mae'].idxmin()
    best_per_config = results_df.loc[idx].reset_index(drop=True)
    
    best_configs_path = os.path.join(output_dir, "best_configs.json")
    best_configs = best_per_config.to_dict(orient='records')
    
    def convert_types(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj
    
    with open(best_configs_path, 'w') as f:
        json.dump(convert_types(best_configs), f, indent=2)
    print(f"\nBest configs saved to {best_configs_path}")
    
    return results_df


def save_results_markdown(results_df: pd.DataFrame, output_dir: str = "evaluation"):
    """Save benchmark results as markdown table."""
    filepath = os.path.join(output_dir, "benchmark_results.md")
    
    with open(filepath, 'w') as f:
        f.write("# Benchmark Results (Per-Ticker Test)\n\n")
        f.write("| Window | Horizon | Model | Ticker | MAE | MSE | Corr |\n")
        f.write("|:------:|:-------:|:------|:-------|----:|----:|-----:|\n")
        
        for window in sorted(results_df['input_window'].unique()):
            window_data = results_df[results_df['input_window'] == window]
            
            for horizon in sorted(window_data['output_horizon'].unique()):
                horizon_data = window_data[window_data['output_horizon'] == horizon]
                
                for _, row in horizon_data.iterrows():
                    mae = f"{row['mae']:.4f}" if not pd.isna(row['mae']) else "N/A"
                    mse = f"{row['mse']:.4f}" if not pd.isna(row['mse']) else "N/A"
                    corr = f"{row.get('corr', 0):.3f}" if not pd.isna(row.get('corr', 0)) else "N/A"
                    f.write(f"| {window} | {horizon} | {row['model']} | {row['ticker']} | {mae} | {mse} | {corr} |\n")
        
        f.write("\n")
    
    print(f"Markdown table saved to {filepath}")


if __name__ == "__main__":
    run_benchmark()
