"""
Directional regime classification and dynamic alpha regeneration trigger.

Regime (pool key): Bull if SMA_20 > SMA_60 else Bear. HighVol (realized_vol_20d
above the trailing-252d 75th percentile) is an orthogonal modifier, not a pool
key: a shift in EITHER direction or volatility state triggers regeneration.

A regime shift is declared when the composite (direction, vol) state persists
for ≥ persist_days consecutive sessions.

DynamicAlphaTrigger is a stateful per-ticker object that decides whether
to regenerate alphas on a given trading day.
"""
import numpy as np
import pandas as pd
from typing import List, Literal, Optional, Tuple

RegimeLabel = Literal['Bull', 'Bear']


def classify_direction(df: pd.DataFrame, idx: int) -> RegimeLabel:
    """Directional regime only (Bull/Bear), independent of volatility."""
    row = df.iloc[idx]
    sma20 = row.get('SMA_20', np.nan)
    sma60 = row.get('SMA_60', np.nan)
    trend = np.sign(sma20 - sma60) if (not np.isnan(sma20) and not np.isnan(sma60)) else 0
    return 'Bull' if trend > 0 else 'Bear'


def is_high_vol(df: pd.DataFrame, idx: int, vol_pct_threshold: int = 75) -> bool:
    """True if realized_vol_20d at idx exceeds the trailing-252d pct threshold."""
    vol_col = 'realized_vol_20d' if 'realized_vol_20d' in df.columns else 'rolling_std_20'
    current_vol = df.iloc[idx].get(vol_col, np.nan)
    start = max(0, idx - 252)
    trailing_vols = df[vol_col].iloc[start:idx + 1].dropna()
    if len(trailing_vols) < 10 or np.isnan(current_vol):
        return False
    return bool(current_vol > np.percentile(trailing_vols.values, vol_pct_threshold))


def detect_regime_shift(
    regime_history: List[RegimeLabel],
    persist_days: int = 5,
) -> bool:
    """
    Return True if the most recent `persist_days` entries in regime_history
    are all identical AND different from the entry immediately preceding them.

    Requires len(regime_history) >= persist_days + 1.
    """
    if len(regime_history) < persist_days + 1:
        return False
    recent = regime_history[-persist_days:]
    if len(set(recent)) != 1:
        return False
    prior = regime_history[-(persist_days + 1)]
    return recent[0] != prior


class DynamicAlphaTrigger:
    """
    Per-ticker state machine that decides when to regenerate alphas.

    Trigger logic (checked daily, in order):
    1. Cooldown: if fewer than cooldown_days have elapsed since the last
       regeneration, skip.
    2. Periodic: if sessions_since_regen >= periodic_freq, regenerate.
    3. Regime shift (if use_regime=True): classify current regime, append
       to history, declare shift if new regime persists >= persist_days and
       differs from previous regime - then regenerate and reset the periodic
       clock.
    """

    def __init__(
        self,
        ticker: str,
        periodic_freq: int = 63,
        use_regime: bool = True,
        cooldown: int = 21,
        persist: int = 5,
        vol_pct_threshold: int = 75,
        ic_decay_threshold: float = 0.02,
    ):
        self.ticker = ticker
        self.periodic_freq = periodic_freq
        self.use_regime = use_regime
        self.cooldown = cooldown
        self.persist = persist
        self.vol_pct_threshold = vol_pct_threshold
        self.ic_decay_threshold = ic_decay_threshold

        self.last_regen_idx: Optional[int] = None
        self.sessions_since_regen: int = 0
        self._regime_history: List[RegimeLabel] = []

    def should_regenerate(
        self,
        current_idx: int,
        df: pd.DataFrame,
        mean_ic: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Check whether alphas should be regenerated at `current_idx`.

        Returns (True, reason) or (False, '').
        Uses df.iloc[:current_idx+1] - no look-ahead.

        Args:
            mean_ic: Spearman IC of the current alpha set over a lagged window.
                     If below ic_decay_threshold, triggers early regeneration.
        """
        # 1. Cooldown guard
        if self.last_regen_idx is not None:
            sessions_elapsed = current_idx - self.last_regen_idx
            if sessions_elapsed < self.cooldown:
                return False, ''

        # 2. IC decay trigger (requires sufficient history; skipped if mean_ic is NaN)
        if mean_ic is not None and not np.isnan(mean_ic):
            if mean_ic < self.ic_decay_threshold:
                return True, 'ic_decay'

        # 3. Periodic trigger
        if self.periodic_freq is not None:
            if self.sessions_since_regen >= self.periodic_freq:
                return True, 'periodic'

        # 4. Regime-shift trigger: composite (direction, volatility) state.
        # A shift in EITHER state triggers regen; the pool key stays directional.
        if self.use_regime:
            d = classify_direction(df, current_idx)
            hv = 'HV' if is_high_vol(df, current_idx, self.vol_pct_threshold) else 'LV'
            regime = f'{d}_{hv}'
            self._regime_history.append(regime)
            # Keep only the last (persist + 1) entries
            if len(self._regime_history) > self.persist + 1:
                self._regime_history = self._regime_history[-(self.persist + 1):]

            if detect_regime_shift(self._regime_history, self.persist):
                return True, 'regime_shift'

        return False, ''

    def mark_regenerated(self, idx: int) -> None:
        """Record that a regeneration occurred at `idx` and reset the clock."""
        self.last_regen_idx = idx
        self.sessions_since_regen = 0

    def tick(self) -> None:
        """Advance the session counter by one (call each day we do NOT regenerate)."""
        self.sessions_since_regen += 1

    def current_regime(self, df: pd.DataFrame, idx: int) -> RegimeLabel:
        """Pool-key regime at idx (no state change): direction only (Bull/Bear).
        Volatility is carried separately via current_high_vol()."""
        return classify_direction(df, idx)

    def current_high_vol(self, df: pd.DataFrame, idx: int) -> bool:
        """Volatility modifier flag (only meaningful in 'directional' mode)."""
        return is_high_vol(df, idx, self.vol_pct_threshold)
