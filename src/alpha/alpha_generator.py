"""
Two-step LLM pipeline for generating formulaic alphas (DA-LLM).

Step 1 (fast model): generate a ≤30-word financial narrative.
Step 2 (strong model): generate 5 alpha expressions per call × 3 calls (max_retries).
Step 3 (C3): filter up to 15 candidates -> top-5 by IC + diversity.
"""
import json
import logging
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .alpha_validator import (
    AlphaValidationError,
    check_sentiment_requirement,
    validate_alpha_expression,
)
from .alpha_filter import select_top_k, _compute_ic as _filter_compute_ic
from .alpha_store import AlphaStore
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


def _compute_ic_for_store(expression: str, df) -> Optional[float]:
    """Compute IC for storage logging; returns None (not NaN) on failure."""
    ic = _filter_compute_ic(expression, df)
    import math
    return None if (ic is None or (isinstance(ic, float) and math.isnan(ic))) else round(float(ic), 4)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_NARRATIVE_TEMPLATE = """\
TICKER: {ticker}
REGIME: {regime}

Last 20 sessions of market data:
{market_table}

252-day summary statistics:
{stats_block}

Write ONE sentence (≤30 words, CFA analyst tone) summarising the current \
market situation.  You MUST mention one technical signal AND one sentiment \
signal.  No preamble, no follow-up - the sentence is the entire response.
"""

_ALPHA_TEMPLATE = """\
You are generating 5 formulaic alpha expressions for {ticker} in {regime} regime.
Generation date: {date}.

IMPORTANT CONTEXT: This is a SINGLE-TICKER time-series problem. There is NO cross-sectional
dimension - you cannot rank this stock against other stocks. All operators are rolling
time-series computations on one stock's own history.

Market narrative: {narrative}

252-day statistics:
{stats_block}

EXPRESSION CONSTRAINTS (strictly enforced):
- Allowed operators: abs, sign, log, rank, delay, delta, ts_rank, ts_mean,
  ts_std, ts_min, ts_max, ts_argmax, ts_argmin, correlation, covariance,
  decay_linear, scale, if_else
- Allowed variables: {allowed_variables}
- Lookback windows d: integers only from the set {{2,3,5,10,15,20,30,60}}
- Maximum nesting depth: 4 levels
- {sentiment_constraint}

SEMANTIC RULES (silent bugs the validator cannot catch):
- Always parenthesize subtraction-then-multiplication. Python evaluates `A - B * C`
  as `A - (B * C)`, not `(A - B) * C`. If you want the latter, write parentheses.
- Do NOT compare 0–100 scaled indicators (RSI_14, BB_upper/middle/lower as price)
  against negative numbers - the condition will never fire.
- Do NOT use a bare boolean comparison (e.g. `X > Y`) as a numeric multiplier.
  Wrap it: `if_else(X > Y, 1, -1)` or `if_else(X > Y, value, 0)`.
- This is single-ticker time-series. Kakushadze's `rank(x)` is cross-sectional and
  has no meaning here. Wherever you would use `rank(x)`, use `ts_rank(x, d)` with an
  explicit window d from the allowed set.
- Do NOT use `sign(<polarity>)` as a multiplier. Polarity is 0 on
  most trading days (sparse news), so `sign(0)=0` collapses the entire
  expression to a constant series -> IC undefined -> alpha rejected. Use polarity
  ADDITIVELY (e.g. `expr + 0.5 * (X_polarity - Y_polarity)`) or via `scale(X_polarity)`.
- Same warning for `if_else(<rare condition>, X, 0)` - if the condition is rarely
  true (e.g. `close > BB_upper` is rare in calm markets), the output is mostly 0
  -> constant series -> rejected. Use a non-zero else branch.

COMPLEXITY REQUIREMENTS (enforced across your 5 alphas):
- Average nesting depth across the 5 expressions should be ≥ 2 (use composed operators,
  not bare `op(var, d)`). Maximum depth 4.
- At least 4 DISTINCT operators must appear across the 5 alphas, drawn from this
  diversity set: {{ts_rank, correlation, covariance, decay_linear, delay, ts_argmax,
  ts_argmin, if_else}}. Do not satisfy this with ts_mean/sign/delta alone.
- At least 5 DISTINCT variables across the 5 alphas. Mix all four families:
  price (open/high/low/close/vwap), volume, technicals (RSI_14, MACD_diff, BB_*,
  EMA_*, SMA_*), and cross-company polarity.
- No two alphas may share the same outermost operator.

STRUCTURAL PATTERNS (Kakushadze-inspired, adapted to single-ticker. These describe
shapes - invent your own concrete expressions for {ticker}):
- Negated rolling rank of a transformed difference: `(-1) * ts_rank(<diff>, d)` where
  <diff> is e.g. delta of a product/sum of two series, optionally combined with an
  additive cross-company polarity term.
- Rank divergence: `ts_rank(<series_A>, d) - ts_rank(<series_B>, d)` for two
  conceptually related series (price vs volume momentum, polarity vs return, etc.).
- Conditional reversal with nested if_else: detect three-state regime
  (extreme up / extreme down / middle) and apply different sign logic to a delta.
- Time-decayed weighting plus a polarity term: `decay_linear(<base_signal>, d)
  + w * (X_polarity - Y_polarity)` or `decay_linear(<base_signal>, d) * scale(X_polarity)`.
- Lead-lag via argmax distance: `ts_argmax(<series_A>, d) - ts_argmax(<series_B>, d)`
  to capture which signal peaked first inside the window.
- Two-window correlation divergence: `correlation(X, Y, short_d) -
  correlation(X, Y, long_d)` to detect short-term shift in co-movement.

{past_performance_block}
Output ONLY a JSON object with this exact schema (5 entries):
{{
  "alphas": [
    {{"expression": "<expression string>", "rationale": "<≤20 word explanation>"}},
    {{"expression": "<expression string>", "rationale": "<≤20 word explanation>"}},
    {{"expression": "<expression string>", "rationale": "<≤20 word explanation>"}},
    {{"expression": "<expression string>", "rationale": "<≤20 word explanation>"}},
    {{"expression": "<expression string>", "rationale": "<≤20 word explanation>"}}
  ]
}}
"""

_TECH_VARS = (
    "open, high, low, close, volume, returns, "
    "SMA_5, SMA_20, SMA_60, "
    "EMA_10, EMA_12, EMA_26, "
    "Momentum_3, Momentum_10, "
    "RSI_14, "
    "MACD, MACD_Signal, MACD_diff, "
    "BB_Upper, BB_Lower, BB_pct_b, BB_width, "
    "OBV, volume_ratio, realized_vol_20d"
)
_ALLOWED_VARS_NO_SENTIMENT = _TECH_VARS
_SENTIMENT_CONSTRAINT_OFF = (
    "Do NOT use any sentiment/polarity variables (no *_polarity columns)."
)


def _build_polarity_constraint(pol_vars: List[str]) -> str:
    """
    Cross-company polarity constraint (paper-2508 style, Table 3). Built per ticker
    from the polarity columns available for that company (target + related firms).
    """
    target = pol_vars[0]
    peer = pol_vars[1] if len(pol_vars) > 1 else pol_vars[0]
    return (
        "At least 1 cross-company polarity variable per alpha, drawn from: "
        + ", ".join(pol_vars)
        + f". Prefer cross-company sentiment divergences between the target and a "
          f"related company, e.g. ({target} - {peer}). Polarity is 0 on days without "
          f"news, so use it ADDITIVELY (e.g. expr + 0.5 * ({target} - {peer})) or via "
          f"scale(); do NOT use sign(<polarity>) as a multiplier."
    )


def _build_past_performance_block(
    past_regime: str,
    past_ic: Optional[float] = None,
    past_alphas_ic: Optional[List[float]] = None,
) -> str:
    """
    Per-alpha IC feedback block (C2).  Lists individual IC for each of the 5
    previous alphas so the LLM can avoid repeating weak patterns.
    Falls back to aggregate IC if per-alpha list is unavailable.
    Returns empty string if no IC data is available.
    """
    if past_alphas_ic is not None and len(past_alphas_ic) > 0:
        lines = [
            f"Prior alpha performance (Spearman IC, 40-session lagged window, "
            f"{past_regime} regime):"
        ]
        for i, ic in enumerate(past_alphas_ic, start=1):
            if np.isnan(ic):
                label = 'N/A'
            else:
                label = 'strong' if ic > 0.05 else ('moderate' if ic > 0.02 else 'weak')
            ic_str = f'{ic:.3f}' if not np.isnan(ic) else 'N/A'
            lines.append(f"  alpha_{i}: IC={ic_str} [{label}]")
        lines.append(
            "Avoid structural patterns shared with weak alphas. "
            "Reinforce patterns from strong ones with fresh variables."
        )
        return '\n'.join(lines) + '\n\n'

    # Fallback: aggregate IC
    if past_ic is not None and not np.isnan(past_ic):
        quality = 'strong' if past_ic > 0.05 else ('moderate' if past_ic > 0.02 else 'weak/decayed')
        return (
            f"Prior alpha set performance (Spearman IC over 40-session lagged window, "
            f"generated in {past_regime} regime): IC = {past_ic:.3f} [{quality}].\n"
            "Generate structurally different alphas to diversify the signal pool.\n\n"
        )
    return ''


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------



def _build_market_table(df: pd.DataFrame, last_n: int = 20) -> str:
    """Format the last N sessions as Z-scored values for the prompt."""
    cols = [
        'close', 'SMA_20', 'SMA_60', 'EMA_12',
        'RSI_14', 'MACD_diff', 'BB_pct_b', 'BB_width', 'volume_ratio',
    ]
    available = [c for c in cols if c in df.columns]
    tail = df[available].tail(last_n)
    mu = df[available].mean()
    sigma = df[available].std().clip(lower=1e-8)
    tail_z = (tail - mu) / sigma
    return tail_z.round(3).to_string(index=False)


def _build_stats_block(df: pd.DataFrame, window: int = 252) -> str:
    """Summarise the trailing window as descriptive statistics."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude legacy generic-sentiment columns and metadata. Per-company polarity
    # columns are kept so the LLM sees their 252-day distribution.
    exclude = {'id', 'ticker'}
    cols = [c for c in numeric_cols if c not in exclude]
    tail = df[cols].tail(window)
    stats = tail.describe().round(4)
    return stats.to_string()


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def _parse_alphas_json(raw: str) -> List[Dict]:
    """Extract the list of alpha dicts from a JSON string."""
    # Strip markdown fences if present
    raw = re.sub(r'```(?:json)?', '', raw).strip('`').strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to find a JSON object within the text
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            raise ValueError(f"Cannot parse JSON from LLM response: {raw[:200]}")

    items = None
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for key in ('alphas', 'results', 'expressions', 'alpha_expressions'):
            if key in data and isinstance(data[key], list):
                items = data[key]
                break
        if items is None:
            for v in data.values():
                if isinstance(v, list):
                    items = v
                    break
    if items is None:
        raise ValueError(f"Unexpected JSON structure: {type(data)}")

    # Normalize: some LLM responses return raw strings instead of dicts
    normalized = []
    for it in items:
        if isinstance(it, str):
            normalized.append({'expression': it, 'rationale': ''})
        elif isinstance(it, dict):
            normalized.append(it)
        # silently skip other types
    return normalized


def _print_alpha_report(
    ticker: str,
    regime: str,
    date: str,
    narrative: str,
    alphas: List[Dict],
    mode: str = 'dynamic',
) -> None:
    label = 'STATIC' if mode == 'static' else 'DYNAMIC'
    print(f"\n{'-' * 40}")
    print(f"  Alpha Generation [{label}] - {ticker} | {regime} | {date}")
    print(f"{'-' * 40}")
    print(f"  Narrative: {narrative}")
    print(f"  {'─'*56}")
    for i, alpha in enumerate(alphas, start=1):
        print(f"  {i}. {alpha.get('expression', '')}")
        rationale = alpha.get('rationale', '').strip()
        if rationale:
            print(f"     to {rationale}")
    print(f"{'-' * 40}\n")


def generate_alphas_for_ticker(
    ticker: str,
    df: pd.DataFrame,
    current_idx: int,
    regime: str,
    client: LLMClient,
    store: AlphaStore,
    with_narrative: bool = True,
    max_retries: int = 3,
    generation_mode: str = 'dynamic',
    use_sentiment: bool = True,
    use_c3_filter: bool = True,
    past_ic: Optional[float] = None,
    past_regime: Optional[str] = None,
    past_alphas_ic: Optional[List[float]] = None,
    trigger_reason: Optional[str] = None,
    sentiment_vars: Optional[List[str]] = None,
    high_vol: bool = False,
) -> List[Dict]:
    """
    Top-level call: generate and validate 5 alphas for `ticker` at `current_idx`.

    C3 strategy: make max_retries calls (each asking for 5 alphas, with increasing
    temperature) to collect up to max_retries*5 diverse candidates, then apply
    select_top_k().  Without C3, stops after first successful call.

    LOOK-AHEAD GUARD: only df.iloc[:current_idx+1] is exposed to the LLM.

    Args:
        with_narrative: If False (ablation A3), skip Step 1 and use an
                        empty narrative placeholder.

    Returns list of dicts with keys: expression, rationale.
    Falls back to the previous valid alpha set on total failure.
    """
    context_df = df.iloc[: current_idx + 1].copy()

    # Directional mode: `regime` is the pool key (Bull/Bear); `high_vol` is the
    # orthogonal volatility modifier shown in the prompt only (not a pool key).
    regime_ctx = f"{regime} (high-volatility)" if high_vol else regime

    # Step 1: narrative
    if with_narrative:
        narrative_prompt = _NARRATIVE_TEMPLATE.format(
            ticker=ticker,
            regime=regime_ctx,
            market_table=_build_market_table(context_df),
            stats_block=_build_stats_block(context_df),
        )
        try:
            narrative = client.call_narrative(narrative_prompt).strip()
        except Exception as e:
            logger.warning(f"[{ticker}] Narrative generation failed: {e}")
            narrative = f"Market in {regime} regime."
    else:
        narrative = f"Market in {regime} regime (no narrative - ablation A3)."

    # Step 2: alpha generation with retry
    stats_block = _build_stats_block(context_df)
    generated_date = str(context_df['date'].iloc[-1].date()) if 'date' in context_df.columns else ''

    # Sentiment vocabulary: build the allowed vars + cross-company polarity
    # constraint per ticker from sentiment_vars. No-sentiment ablation drops them.
    if use_sentiment and sentiment_vars:
        allowed_vars = _TECH_VARS + ", " + ", ".join(sentiment_vars)
        sent_constraint = _build_polarity_constraint(sentiment_vars)
    else:
        allowed_vars = _ALLOWED_VARS_NO_SENTIMENT
        sent_constraint = _SENTIMENT_CONSTRAINT_OFF

    alpha_prompt = _ALPHA_TEMPLATE.format(
        ticker=ticker,
        regime=regime_ctx,
        date=generated_date,
        narrative=narrative,
        stats_block=stats_block,
        allowed_variables=allowed_vars,
        sentiment_constraint=sent_constraint,
        past_performance_block=_build_past_performance_block(
            past_regime=past_regime or regime,
            past_ic=past_ic,
            past_alphas_ic=past_alphas_ic,
        ),
    )

    # C3: collect ALL valid candidates across retries, then filter top-5
    all_valid: List[Dict] = []

    for attempt in range(max_retries):
        temperature = 0.7 + attempt * 0.1
        try:
            raw = client.call_alpha(alpha_prompt, temperature=temperature, use_cache=(attempt == 0))
            candidates = _parse_alphas_json(raw)
        except Exception as e:
            logger.warning(f"[{ticker}] Alpha parse failed (attempt {attempt + 1}): {e}")
            continue

        for alpha in candidates:
            expr = alpha.get('expression', '').strip()
            if not expr:
                continue
            try:
                validate_alpha_expression(expr)
                has_sentiment = check_sentiment_requirement(expr)
                if not use_sentiment and has_sentiment:
                    raise AlphaValidationError(
                        f"Sentiment variable in no-sentiment mode: {expr}"
                    )
                alpha['is_valid'] = 1
                if use_sentiment and not has_sentiment and 'sentiment' not in alpha.get('rationale', '').lower():
                    logger.debug(f"[{ticker}] Alpha missing sentiment: {expr}")
                all_valid.append(alpha)
            except AlphaValidationError as ve:
                logger.warning(f"[{ticker}] Invalid alpha expression: {expr} - {ve}")

        # C3: collect all 3 rounds (5 per round × 3 = up to 15 candidates).
        # Without C3: stop as soon as we have 5.
        target = max_retries * 5 if use_c3_filter else 5
        if len(all_valid) >= target:
            break

    if len(all_valid) < 5:
        logger.warning(
            f"[{ticker}] Only {len(all_valid)} valid alphas after {max_retries} attempts. "
            "Falling back to previous alpha set."
        )
        prior = store.get_active_alphas_v2(ticker, as_of_idx=current_idx)
        if prior:
            all_valid = [
                {'expression': e, 'rationale': 'carried over', 'is_valid': 1}
                for e in prior[:5]
            ]
        if len(all_valid) == 0:
            logger.error(f"[{ticker}] No valid alphas available. Returning empty list.")
            return []

    # C3: select top-5 by IC + diversity from all valid candidates
    # Ablation A3 (use_c3_filter=False): skip filter, take first 5 by validation order
    n_candidates = len(all_valid)
    if use_c3_filter and n_candidates > 5:
        validated_alphas, selected_ics = select_top_k(all_valid, context_df, k=5)
    else:
        validated_alphas = all_valid[:5]
        selected_ics = [_compute_ic_for_store(a['expression'], context_df) for a in validated_alphas]

    alphas_to_store = validated_alphas[:5]
    store.store_generation(
        ticker=ticker,
        generated_at_idx=current_idx,
        generated_at_date=generated_date,
        regime=regime,
        narrative=narrative,
        alphas=alphas_to_store,
        generation_mode=generation_mode,
        trigger_reason=trigger_reason,
        alpha_ics=selected_ics,
        n_candidates=n_candidates,
    )

    logger.info(f"[{ticker}] Generated {len(alphas_to_store)} alphas at idx={current_idx}, regime={regime}")
    _print_alpha_report(ticker, regime, generated_date, narrative, alphas_to_store, generation_mode)
    return alphas_to_store
