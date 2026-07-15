"""
Walk-forward orchestrator for DA-LLM.

For each fold, one model is trained per ticker (paper Algorithm 1). Evaluation
produces one FoldResult per (ticker, fold, model, horizon, seed).

DA-LLM alpha pre-computation:
    Before the fold loop, the orchestrator simulates the time series day by day,
    generating alpha expressions whenever DynamicAlphaTrigger fires (C1).
    Generation uses per-alpha IC feedback (C2) and generate-and-filter top-k (C3).
    Alpha values are stored in columns alpha_1..5 - no look-ahead because the
    trigger and generator both receive only df.iloc[:idx+1].

Checkpointing: maps (fold_idx, ticker) to List[FoldResult].
"""
import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.data_utils import (
    load_stock_data,
    ALPHA_COLS,
    METHOD_M_COLS,
)

from ..features.sentiment import load_polarity_features
from ..features.alpha_executor import evaluate_alpha_expression


def _compute_per_alpha_ic(
    expressions: List[str],
    df: pd.DataFrame,
    current_idx: int,
    window: int = 40,
    lag: int = 1,
) -> List[float]:
    """
    Per-alpha Spearman IC vs next-day returns over a lagged window (C2).

    Returns a list of length len(expressions), with np.nan for alphas that
    could not be evaluated or had insufficient data.
    lag=1 ensures current day's return is excluded - no leakage.
    """
    ret_col = 'log_return' if 'log_return' in df.columns else 'returns'
    if not expressions or ret_col not in df.columns:
        return [np.nan] * len(expressions)

    alpha_end = current_idx - lag
    alpha_start = alpha_end - window
    if alpha_start < 1:
        return [np.nan] * len(expressions)

    context = df.iloc[:alpha_end]
    ret_values = df[ret_col].iloc[alpha_start + 1: alpha_end + 1].values

    ics: List[float] = []
    for expr in expressions:
        try:
            alpha_series = evaluate_alpha_expression(expr, context, validate=False)
        except Exception:
            ics.append(np.nan)
            continue
        alpha_values = alpha_series.iloc[alpha_start:alpha_end].values
        mask = ~(np.isnan(alpha_values) | np.isnan(ret_values))
        if mask.sum() < window // 2:
            ics.append(np.nan)
            continue
        ic = pd.Series(alpha_values[mask]).corr(pd.Series(ret_values[mask]), method='spearman')
        ics.append(float(ic) if not np.isnan(ic) else np.nan)

    return ics


def _compute_rolling_spearman_ic(
    expressions: List[str],
    df: pd.DataFrame,
    current_idx: int,
    window: int = 40,
    lag: int = 1,
) -> float:
    """Mean Spearman IC - used by trigger logic only."""
    ics = [x for x in _compute_per_alpha_ic(expressions, df, current_idx, window, lag)
           if not np.isnan(x)]
    return float(np.mean(ics)) if ics else np.nan


from ..alpha.alpha_store import AlphaStore
from ..alpha.llm_client import LLMClient
from ..alpha.alpha_generator import generate_alphas_for_ticker
from ..alpha.regime_detector import DynamicAlphaTrigger
from .fold_generator import Fold
from .fold_runner import run_fold, FoldResult

logger = logging.getLogger(__name__)


class WalkForwardOrchestrator:
    def __init__(self, config: dict, logger_: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger_ or logger

        data_cfg = config.get('data', {})
        self._tickers: List[str] = data_cfg.get('tickers', [])
        self._price_dir: str = data_cfg.get('price_dir', 'data/price/csv')
        self._polarity_features_dir: str = data_cfg.get('polarity_features_dir', 'data/news/sentiment_features')
        self._alphas_dir: str = data_cfg.get('alphas_dir', 'data/alphas')
        self._horizons: List[int] = data_cfg.get('horizons', [1, 5])
        self._start_date: Optional[str] = data_cfg.get('start_date')
        self._end_date: Optional[str] = data_cfg.get('end_date')
        self._target_col: str = data_cfg.get('target_col', 'close')

        regen_cfg = config.get('alpha_regen', {})
        self._periodic_freq = regen_cfg.get('periodic_frequency', 63)
        self._use_regime = regen_cfg.get('use_regime', True)
        self._use_sentiment = data_cfg.get('use_sentiment', True)
        self._with_narrative: bool = config.get('llm', {}).get('with_narrative', True)
        self._cooldown = regen_cfg.get('cooldown_days', 21)
        self._regime_persist = regen_cfg.get('regime_persist_days', 5)
        self._vol_pct = regen_cfg.get('vol_percentile_threshold', 75)

        # C3: generate 15 candidates and filter top-5 by IC + diversity
        self._use_c3_filter: bool = config.get('use_c3_filter', True)
        # C2: per-alpha IC feedback in LLM prompt (ablation A2 sets this False)
        self._use_c2_feedback: bool = config.get('use_c2_feedback', True)

        training_cfg = config.get('training', {})
        self._seeds: List[int] = training_cfg.get('seeds', [42])

        self._model_types: List[str] = list(config.get('models', {}).keys())

        # Use a checkpoint dir derived from alphas_dir so each ablation
        # gets its own checkpoint and doesn't reuse another config's results.
        alphas_tag = self._alphas_dir.replace('/', '_').strip('_')
        self._results_dir = os.path.join('results/raw', alphas_tag)
        os.makedirs(self._results_dir, exist_ok=True)

        # Shared resources
        self._store: Optional[AlphaStore] = None
        self._llm_client: Optional[LLMClient] = None

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _get_store(self) -> AlphaStore:
        if self._store is None:
            self._store = AlphaStore(self._alphas_dir)
        return self._store

    def _get_llm_client(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = LLMClient(self.config.get('llm', {}))
        return self._llm_client


    # ------------------------------------------------------------------
    # Enriched DataFrame builder (per ticker)
    # ------------------------------------------------------------------

    def prepare_enriched_dataframe(
        self,
        ticker: str,
        folds: Optional[List[Fold]] = None,
        with_narrative: bool = True,
        use_regime: bool = True,
        use_sentiment: bool = True,
        periodic_freq: Optional[int] = None,
        use_c3_filter: bool = True,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Build the feature-enriched DataFrame for one ticker (DA-LLM)."""
        df = load_stock_data(ticker, self._start_date, self._end_date)
        if df is None:
            raise FileNotFoundError(f"No price data for {ticker}")

        df = df.sort_values('date').reset_index(drop=True)

        polarity_df = load_polarity_features(ticker, self._polarity_features_dir)
        if polarity_df is not None:
            df = df.merge(polarity_df, on='date', how='left')
            pol_cols = [c for c in polarity_df.columns if c != 'date']
            df[pol_cols] = df[pol_cols].fillna(0.0)

        if folds is None:
            raise ValueError("folds must be provided")

        if not use_sentiment:
            for col in ['avg_sentiment', 'news_count', 'has_news',
                        'sentiment_ma5', 'sentiment_ma20', 'sentiment_momentum']:
                df[col] = 0.0

        pfreq = periodic_freq if periodic_freq is not None else self._periodic_freq
        df = self.precompute_method_m_alphas(
            ticker=ticker,
            df_base=df,
            folds=folds,
            periodic_freq=pfreq,
            use_regime=use_regime,
            with_narrative=with_narrative,
            use_sentiment=use_sentiment,
            use_c3_filter=use_c3_filter,
        )
        return df, list(METHOD_M_COLS)

    def _get_regime_at(self, df: pd.DataFrame, idx: int) -> str:
        from ..alpha.regime_detector import classify_regime
        return classify_regime(df, idx, self._vol_pct)

    def precompute_method_m_alphas(
        self,
        ticker: str,
        df_base: pd.DataFrame,
        folds: List[Fold],
        periodic_freq: int = 63,
        use_regime: bool = True,
        with_narrative: bool = True,
        use_sentiment: bool = True,
        use_c3_filter: bool = True,
    ) -> pd.DataFrame:
        """
        Simulate time-series day by day; generate dynamic alphas on trigger events.
        This is the DA-LLM dynamic regeneration path (C1+C2+C3).
        """
        df = df_base.copy()
        for col in ALPHA_COLS:
            df[col] = np.nan

        if not folds:
            return df

        wf_start_idx = 0
        # Cap regeneration at the FIRST fold's test_start so no alpha is selected
        # using test-period returns. For walk-forward, this means alphas regenerate
        # only during initial train+val; for single split, only inside train+val.
        # Test-window rows still receive alpha values via the post-loop materialisation
        # below, but the *expressions themselves* are frozen at test_start.
        wf_end_idx = folds[0].test_start

        trigger = DynamicAlphaTrigger(
            ticker=ticker,
            periodic_freq=periodic_freq,
            use_regime=use_regime,
            cooldown=self._cooldown,
            persist=self._regime_persist,
            vol_pct_threshold=self._vol_pct,
        )

        store = self._get_store()
        client = self._get_llm_client()

        # Paper-2508 sentiment: expose this ticker's per-company polarity columns
        # (target + related firms) to the alpha generator.
        sentiment_vars = [c for c in df.columns if c.endswith('_polarity')]
        if not sentiment_vars:
            logger.warning(f"[{ticker}] no *_polarity columns found for alpha generation.")

        wf_end = min(wf_end_idx, len(df))

        trigger_events: List[Tuple[int, List[str], str]] = []

        for idx in range(wf_start_idx, wf_end):
            current_exprs = trigger_events[-1][1] if trigger_events else []
            # C2: compute per-alpha IC for individual feedback in the LLM prompt
            per_alpha_ics = _compute_per_alpha_ic(current_exprs, df, idx) if current_exprs else []
            mean_ic = float(np.nanmean(per_alpha_ics)) if per_alpha_ics else np.nan

            should_regen, reason = trigger.should_regenerate(idx, df, mean_ic=mean_ic)

            if should_regen:
                regime = trigger.current_regime(df, idx)
                high_vol = trigger.current_high_vol(df, idx)
                past_regime = trigger_events[-1][2] if trigger_events else None

                # Fast path: reuse already-stored alphas (avoids re-calling the LLM)
                cached_exprs = store.get_active_alphas_v2(ticker, as_of_idx=idx + 1)
                if cached_exprs and store._has_generation_at(ticker, idx):
                    new_exprs = cached_exprs[:10]
                    trigger.mark_regenerated(idx)
                    trigger_events.append((idx, new_exprs, regime))
                    logger.debug(f"[{ticker}] Reusing cached alphas at idx={idx}")
                else:
                    try:
                        new_alphas = generate_alphas_for_ticker(
                            ticker=ticker,
                            df=df,
                            current_idx=idx,
                            regime=regime,
                            client=client,
                            store=store,
                            with_narrative=with_narrative,
                            use_sentiment=use_sentiment,
                            past_ic=mean_ic,
                            past_regime=past_regime,
                            past_alphas_ic=per_alpha_ics if (per_alpha_ics and self._use_c2_feedback) else None,
                            use_c3_filter=use_c3_filter,
                            trigger_reason=reason,
                            sentiment_vars=sentiment_vars,
                            high_vol=high_vol,
                        )
                        new_exprs = [a['expression'] for a in new_alphas]
                        trigger.mark_regenerated(idx)
                        trigger_events.append((idx, new_exprs[:10], regime))
                        logger.info(
                            f"[{ticker}] Regenerated at idx={idx}, reason={reason}, "
                            f"regime={regime}, n_alphas={len(new_exprs)}, mean_ic={mean_ic:.3f}"
                        )
                    except Exception as e:
                        logger.error(f"[{ticker}] Alpha generation failed at idx={idx}: {e}")
                        trigger.tick()
            else:
                trigger.tick()

        # Materialise alpha values. The last trigger event's expressions remain
        # active through the test window (frozen - not re-selected on test returns).
        timeline_end = len(df)
        for t_i, (start_idx, exprs, _regime) in enumerate(trigger_events):
            end_idx = trigger_events[t_i + 1][0] if t_i + 1 < len(trigger_events) else timeline_end
            idx_labels = df.index[start_idx:end_idx]
            for j, expr in enumerate(exprs, start=1):
                col = f'alpha_{j}'
                series = evaluate_alpha_expression(expr, df, validate=False)
                df.loc[idx_labels, col] = series.loc[idx_labels].values

        # Normalize alpha columns to stabilize scale across dynamic expressions.
        # Different expressions output wildly different ranges; rolling z-score
        # (window=252, clipped to [-3,3]) brings all alpha columns to the same scale
        # before MinMaxScaler is applied downstream.
        alpha_cols = [f'alpha_{j}' for j in range(1, 11)]
        for col in alpha_cols:
            if col not in df.columns:
                continue
            mu = df[col].rolling(252, min_periods=20).mean()
            sigma = df[col].rolling(252, min_periods=20).std().clip(lower=1e-8)
            df[col] = ((df[col] - mu) / sigma).clip(-3, 3)

        return df

    # ------------------------------------------------------------------
    # Standalone alpha generation (no training)
    # ------------------------------------------------------------------

    def generate_alphas_only(
        self,
        tickers: Optional[List[str]] = None,
    ) -> None:
        tickers = tickers or self._tickers
        if not tickers:
            raise ValueError("No tickers specified.")

        folds = self._generate_single_fold(tickers)
        if folds is None:
            raise RuntimeError("Could not generate folds - check price data.")

        for ticker in tickers:
            df = load_stock_data(ticker, self._start_date, self._end_date)
            if df is None:
                logger.warning(f"[{ticker}] No price data found, skipping.")
                continue
            df = df.sort_values('date').reset_index(drop=True)

            polarity_df = load_polarity_features(ticker, self._polarity_features_dir)
            if polarity_df is not None:
                df = df.merge(polarity_df, on='date', how='left')
                pol_cols = [c for c in polarity_df.columns if c != 'date']
                df[pol_cols] = df[pol_cols].fillna(0.0)

            print(f"\n[{ticker}] DA-LLM alpha generation starting …")
            try:
                self.precompute_method_m_alphas(
                    ticker=ticker,
                    df_base=df,
                    folds=folds,
                    periodic_freq=self._periodic_freq,
                    use_regime=self._use_regime,
                    with_narrative=self._with_narrative,
                    use_sentiment=self._use_sentiment,
                    use_c3_filter=self._use_c3_filter,
                )
                store = self._get_store()
                n_events = len(store.get_all_generations(ticker))
                print(f"[{ticker}] DA-LLM done - {n_events} generation event(s) stored.")
            except Exception as e:
                logger.error(f"[{ticker}] DA-LLM generation failed: {e}")

        client = self._get_llm_client()
        if client.usage:
            print("\n=== LLM Token Usage ===")
            for model, u in client.usage.items():
                print(f"  {model}: {u['in']:,} in + {u['out']:,} out = {u['in']+u['out']:,} total")
            print("=======================\n")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _checkpoint_path(self) -> str:
        return os.path.join(self._results_dir, "method_m_checkpoint.pkl")

    def _load_checkpoint(self, path: str) -> Optional[Dict[int, List[FoldResult]]]:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None

    def _save_checkpoint(self, path: str, completed: Dict[int, List[FoldResult]]) -> None:
        with open(path, 'wb') as f:
            pickle.dump(completed, f)

    def _generate_single_fold(self, tickers: List[str]) -> Optional[List[Fold]]:
        """
        Generate ONE fold representing a 70/30 chronological split (paper §3).

        Train  : rows [0, 0.70 * n)        (70% - matches paper exactly)
        Val    : empty                      (paper has no validation set; trains
                                             fixed num_epochs and uses final model)
        Test   : rows [0.70*n, n)           (30%)

        Uses the minimum session count across tickers for fold boundary alignment.
        """
        session_counts = {}
        for ticker in tickers:
            df = load_stock_data(ticker, self._start_date, self._end_date)
            if df is None:
                continue
            session_counts[ticker] = len(df)

        if not session_counts:
            return None

        min_ticker = min(session_counts, key=session_counts.get)
        n = session_counts[min_ticker]

        train_end  = int(0.70 * n)
        test_end   = n

        logger.info(
            f"Single 70/30 split (paper §3, no val): n={n} (min ticker={min_ticker}) | "
            f"train=[0,{train_end}) test=[{train_end},{test_end})"
        )

        return [Fold(
            fold_idx=0,
            train_start=0,
            train_end=train_end,
            val_start=train_end,
            val_end=train_end,    # empty val
            test_start=train_end,
            test_end=test_end,
        )]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_all(
        self,
        model_types: Optional[List[str]] = None,
        tickers: Optional[List[str]] = None,
        split: str = 'single',
        save_models_dir: Optional[str] = None,
    ) -> List[FoldResult]:
        """
        Run the DA-LLM experiment grid (one model per ticker, paper Algorithm 1).

        Args:
            split: retained for compatibility; the run always uses the paper's
                70/30 single split.
            save_models_dir: if set, per-(ticker, seed) state_dict + history JSON
                are saved under <save_models_dir>/<model_type>/<TICKER>/.
        """
        model_types = model_types or self._model_types
        tickers     = tickers     or self._tickers

        all_results: List[FoldResult] = []

        ckpt_path = self._checkpoint_path()
        completed: Dict = self._load_checkpoint(ckpt_path) or {}

        for _, cached_results in completed.items():
            if isinstance(cached_results, list):
                all_results.extend(cached_results)

        folds = self._generate_single_fold(tickers)

        if folds is None:
            logger.error("Could not generate folds")
            return all_results

        # Build enriched DataFrames for all tickers
        enriched_dfs: Dict[str, pd.DataFrame] = {}
        ticker_feature_cols: Dict[str, List[str]] = {}
        for ticker in tickers:
            try:
                df_e, cols = self.prepare_enriched_dataframe(
                    ticker=ticker, folds=folds,
                    use_regime=self._use_regime,
                    use_sentiment=self._use_sentiment,
                    with_narrative=self._with_narrative,
                    use_c3_filter=self._use_c3_filter,
                )
                enriched_dfs[ticker] = df_e
                ticker_feature_cols[ticker] = cols
            except Exception as e:
                logger.error(f"[{ticker}] enrichment failed: {e}")

        if not enriched_dfs:
            logger.error("No enriched data available")
            return all_results

        for fold in folds:
            self._run_fold_per_ticker(
                fold=fold, model_types=model_types,
                enriched_dfs=enriched_dfs, ticker_feature_cols=ticker_feature_cols,
                completed=completed, ckpt_path=ckpt_path, all_results=all_results,
                save_models_dir=save_models_dir,
            )

        return all_results

    def _run_fold_per_ticker(
        self,
        fold: 'Fold',
        model_types: List[str],
        enriched_dfs: Dict[str, pd.DataFrame],
        ticker_feature_cols: Dict[str, List[str]],
        completed: Dict,
        ckpt_path: str,
        all_results: List[FoldResult],
        save_models_dir: Optional[str] = None,
    ) -> None:
        """Train one model per ticker for a given fold (paper Algorithm 1)."""
        for ticker, df_enriched in enriched_dfs.items():
            ck_key = (fold.fold_idx, ticker)
            if ck_key in completed:
                continue

            feature_cols = ticker_feature_cols[ticker]
            ticker_results: List[FoldResult] = []

            for model_type in model_types:
                model_save_dir = (
                    os.path.join(save_models_dir, model_type)
                    if save_models_dir else None
                )
                for horizon in self._horizons:
                    for seed in self._seeds:
                        try:
                            result = run_fold(
                                fold=fold,
                                ticker=ticker,
                                model_type=model_type,
                                horizon=horizon,
                                seed=seed,
                                df_enriched=df_enriched,
                                feature_cols=feature_cols,
                                config=self.config,
                                save_dir=model_save_dir,
                            )
                            ticker_results.append(result)
                        except Exception as e:
                            logger.error(
                                f"[{ticker}] fold={fold.fold_idx} "
                                f"{model_type} h={horizon} seed={seed}: {e}",
                                exc_info=True,
                            )

            completed[ck_key] = ticker_results
            self._save_checkpoint(ckpt_path, completed)
            all_results.extend(ticker_results)
            logger.info(
                f"[{ticker}] fold={fold.fold_idx}: "
                f"{len(ticker_results)} results, checkpoint saved"
            )
