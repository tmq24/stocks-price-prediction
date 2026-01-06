import numpy as np
def add_features(df):
    df = df.sort_values('date').copy()

    # =====================
    # Moving Averages
    # =====================
    df['SMA_5'] = df['close'].rolling(5).mean()
    df['SMA_20'] = df['close'].rolling(20).mean()
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()

    # =====================
    # Momentum
    # =====================
    df['Momentum_3'] = df['close'] - df['close'].shift(3)
    df['Momentum_10'] = df['close'] - df['close'].shift(10)

    # =====================
    # RSI 14
    # =====================
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # =====================
    # MACD
    # =====================
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # =====================
    # Bollinger Bands
    # =====================
    rolling_std = df['close'].rolling(20).std()
    df['BB_Upper'] = df['SMA_20'] + 2 * rolling_std
    df['BB_Lower'] = df['SMA_20'] - 2 * rolling_std

    # =====================
    # OBV – FIX ĐÚNG
    # =====================
    direction = np.sign(df['close'].diff()).fillna(0)
    df['OBV'] = (direction * df['volume']).cumsum()

    return df