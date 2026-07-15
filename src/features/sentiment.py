import os
import pandas as pd
from typing import Optional

POLARITY_FEATURES_DIR = 'data/news/sentiment_features'


def load_polarity_features(
    ticker: str,
    features_dir: str = POLARITY_FEATURES_DIR,
) -> Optional[pd.DataFrame]:
    """
    Load per-company polarity features from sentiment_features/<TICKER>.csv.

    All *_polarity columns are shifted by 1 day (news on day t to feature for
    day t+1), matching the blanket shift in compute_all_indicators().
    polarity_avg = mean of all *_polarity columns is pre-computed for use in
    TCEHY alpha expressions where the paper averages 12 company polarities.

    Returns None if the file does not exist.
    """
    path = os.path.join(features_dir, f'{ticker}.csv')
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])

    polarity_cols = [c for c in df.columns if c.endswith('_polarity')]
    df[polarity_cols] = df[polarity_cols].fillna(0.0)

    if polarity_cols:
        df['polarity_avg'] = df[polarity_cols].mean(axis=1)
        all_shift_cols = polarity_cols + ['polarity_avg']
    else:
        all_shift_cols = []

    for col in all_shift_cols:
        df[col] = df[col].shift(1)

    return df[['date'] + all_shift_cols]
