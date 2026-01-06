import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

# ------------------------------------------------
# 1. Xác định feature columns (Alpha + Technical)
# ------------------------------------------------
def get_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude_cols = [
        'ticker', 'date', 'adjClose', 'close',
        'day_idx', 'week_idx', 'month_idx'
    ]
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    return [c for c in numeric_cols if c not in exclude_cols]


# ------------------------------------------------
# 2. Scale FEATURE (X)
# ------------------------------------------------
def scale_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    is_train: bool,
    scaler: StandardScaler = None
) -> Tuple[pd.DataFrame, StandardScaler]:

    df_scaled = df.copy()

    if is_train:
        scaler = StandardScaler()
        df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])
        print(f"✓ TRAIN: Scaled {len(feature_cols)} feature columns")
    else:
        df_scaled[feature_cols] = scaler.transform(df_scaled[feature_cols])
        print("✓ TEST: Applied feature scaler")

    return df_scaled, scaler


# ------------------------------------------------
# 3. Scale LABEL (adjClose)
# ------------------------------------------------
def scale_label(
    df: pd.DataFrame,
    label_col: str = 'adjClose',
    is_train: bool = True,
    scaler: StandardScaler = None
) -> Tuple[pd.DataFrame, StandardScaler]:

    df_scaled = df.copy()

    if is_train:
        scaler = StandardScaler()
        df_scaled[[label_col]] = scaler.fit_transform(df_scaled[[label_col]])
        print("✓ TRAIN: Scaled adjClose label")
    else:
        df_scaled[[label_col]] = scaler.transform(df_scaled[[label_col]])
        print("✓ TEST: Applied label scaler")

    return df_scaled, scaler


# ------------------------------------------------
# 4. FULL PIPELINE
# ------------------------------------------------
def full_scale_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
):
    # Feature columns
    feature_cols = get_feature_columns(train_df)

    # Scale features
    train_feat_scaled, x_scaler = scale_features(
        train_df, feature_cols, is_train=True
    )
    test_feat_scaled, _ = scale_features(
        test_df, feature_cols, is_train=False, scaler=x_scaler
    )

    # Scale label
    train_final, y_scaler = scale_label(
        train_feat_scaled, 'adjClose', is_train=True
    )
    test_final, _ = scale_label(
        test_feat_scaled, 'adjClose', is_train=False, scaler=y_scaler
    )

    return train_final, test_final, feature_cols, x_scaler, y_scaler