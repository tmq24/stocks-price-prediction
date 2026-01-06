import matplotlib.pyplot as plt
import pandas as pd

def plot_predictions_real(
    preds,
    trues,
    df_test_raw,
    window_size=5
):
    TICKER_COL = "ticker"

    df_sorted = df_test_raw.sort_values([TICKER_COL, "date"]).reset_index(drop=True)

    dates, tickers = [], []

    for ticker, group in df_sorted.groupby(TICKER_COL):
        dates.extend(group["date"].iloc[window_size:].tolist())
        tickers.extend([ticker] * (len(group) - window_size))

    plot_df = pd.DataFrame({
        "Date": dates,
        "Ticker": tickers,
        "Actual": trues,
        "Predicted": preds
    })

    num_tickers = plot_df["Ticker"].nunique()
    plt.figure(figsize=(16, 5 * num_tickers))

    for i, ticker in enumerate(plot_df["Ticker"].unique()):
        data = plot_df[plot_df["Ticker"] == ticker]

        plt.subplot(num_tickers, 1, i + 1)
        plt.plot(data["Date"], data["Actual"], label="Actual", linewidth=2)
        plt.plot(data["Date"], data["Predicted"], "--", label="Predicted", linewidth=1.5)

        plt.title(f"{ticker} – Predict adjClose (Next Day)")
        plt.ylabel("Price")
        plt.xlabel("Date")
        plt.legend()
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()