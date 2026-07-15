"""JSON-backed persistence for generated alpha expressions.

One file per ticker: {store_dir}/{ticker}.json
Each file is a list of generation events (newest last), making it easy
to open and read without any database tooling.

Example file structure (data/alphas/AAPL.json):
[
  {
    "generated_at_idx": 63,
    "generated_at_date": "2021-03-15",
    "regime": "Bull",
    "narrative": "RSI signals oversold conditions ...",
    "generation_mode": "dynamic",
    "created_ts": "2024-01-01T00:00:00+00:00",
    "alphas": [
      {"rank": 1, "expression": "ts_mean(close, 20) - ts_mean(close, 60)", "rationale": "...", "is_valid": 1},
      ...
    ]
  },
  ...
]
"""
import json
import os
from datetime import datetime, timezone
from typing import List, Optional, Tuple


class AlphaStore:
    def __init__(self, store_dir: str):
        os.makedirs(store_dir, exist_ok=True)
        self._dir = store_dir

    def _path(self, ticker: str) -> str:
        return os.path.join(self._dir, f"{ticker}.json")

    def _load(self, ticker: str) -> list:
        path = self._path(ticker)
        if not os.path.exists(path):
            return []
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save(self, ticker: str, events: list) -> None:
        with open(self._path(ticker), 'w', encoding='utf-8') as f:
            json.dump(events, f, indent=2, ensure_ascii=False)

    def store_generation(
        self,
        ticker: str,
        generated_at_idx: int,
        generated_at_date: str,
        regime: str,
        narrative: Optional[str],
        alphas: List[dict],
        generation_mode: str = 'dynamic',
        trigger_reason: Optional[str] = None,
        alpha_ics: Optional[List[float]] = None,
        n_candidates: int = 0,
    ) -> None:
        """
        trigger_reason: 'periodic', 'regime', 'ic_decay', or None
        alpha_ics: per-alpha IC of the selected top-5 (for C2/C3 analysis)
        n_candidates: total valid candidates before C3 filtering
        """
        events = self._load(ticker)
        if any(e['generated_at_idx'] == generated_at_idx and e.get('generation_mode') == generation_mode for e in events):
            return  # idempotent: skip duplicate for same idx + mode
        ics = alpha_ics or [None] * len(alphas)
        def _clean(v):
            """Convert numpy floats and NaN to JSON-safe scalars."""
            if v is None:
                return None
            try:
                f = float(v)
                if f != f:  # NaN check
                    return None
                return f
            except (TypeError, ValueError):
                return None

        events.append({
            'generated_at_idx': generated_at_idx,
            'generated_at_date': generated_at_date,
            'regime': regime,
            'trigger_reason': trigger_reason,
            'narrative': narrative,
            'generation_mode': generation_mode,
            'n_candidates': n_candidates,
            'created_ts': datetime.now(timezone.utc).isoformat(),
            'alphas': [
                {
                    'rank': i + 1,
                    'expression': a.get('expression', ''),
                    'rationale': a.get('rationale', ''),
                    'is_valid': int(a.get('is_valid', 1)),
                    'ic': _clean(ics[i] if i < len(ics) else None),
                    # C3 metadata (split-window OOS validation, written by alpha_filter.select_top_k)
                    'ic_select': _clean(a.get('_ic_select')),
                    'ic_val':    _clean(a.get('_ic_val')),
                    'score':     _clean(a.get('_score')),
                }
                for i, a in enumerate(alphas)
            ],
        })
        self._save(ticker, events)

    def get_active_alphas_v2(self, ticker: str, as_of_idx: int) -> List[str]:
        """Return the 5 expressions from the latest event before as_of_idx."""
        prior = [e for e in self._load(ticker) if e['generated_at_idx'] < as_of_idx]
        if not prior:
            return []
        latest = max(prior, key=lambda e: e['generated_at_idx'])
        return [
            a['expression']
            for a in sorted(latest['alphas'], key=lambda a: a['rank'])
            if a['is_valid'] == 1
        ][:5]

    def _has_generation_at(self, ticker: str, idx: int) -> bool:
        """Return True if there's a dynamic generation stored at exactly this idx."""
        return any(
            e['generated_at_idx'] == idx and e.get('generation_mode', 'dynamic') == 'dynamic'
            for e in self._load(ticker)
        )

    def get_latest_generation_idx(self, ticker: str) -> Optional[int]:
        events = self._load(ticker)
        return max((e['generated_at_idx'] for e in events), default=None)

    def get_all_generations(self, ticker: str) -> List[Tuple[int, str]]:
        return [(e['generated_at_idx'], e['regime']) for e in self._load(ticker)]

    def flag_low_quality(self, ticker: str, generated_at_idx: int, alpha_rank: int) -> None:
        """Mark an alpha as low-quality (is_valid=2). Still included in experiments."""
        events = self._load(ticker)
        for e in events:
            if e['generated_at_idx'] == generated_at_idx:
                for a in e['alphas']:
                    if a['rank'] == alpha_rank:
                        a['is_valid'] = 2
        self._save(ticker, events)
