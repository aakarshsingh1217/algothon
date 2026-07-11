"""Reusable indicators for Algothon notebook analysis."""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 250


def load_prices(path="prices.txt"):
    """Load prices.txt as a DataFrame (rows=days, columns=tickers)."""
    return pd.read_csv(path, sep=r"\s+", header=0)


def log_returns(prices):
    """Daily log returns from price levels."""
    return np.log(prices / prices.shift(1))


def rsi(prices, window=14):
    """
    Relative Strength Index (0-100).

    Accepts a Series or DataFrame of prices.
    """
    if isinstance(prices, pd.DataFrame):
        return prices.apply(lambda col: rsi(col, window=window))

    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def sharpe_ratio(returns, periods_per_year=TRADING_DAYS_PER_YEAR):
    """
    Annualized Sharpe ratio from daily returns or daily PnL.

    Uses the same convention as eval.py: sqrt(250) * mean / std.
    """
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return np.nan

    std = r.std()
    if std < 1e-10:
        return np.nan

    return float(np.sqrt(periods_per_year) * r.mean() / std)


def sharpe_ratio_per_instrument(prices, last_n_days=None):
    """
    Sharpe ratio for each ticker from price levels.

    Set last_n_days=250 to match eval.py's test window.
    """
    if last_n_days is not None:
        prices = prices.iloc[-last_n_days:]

    rets = log_returns(prices)
    return rets.apply(sharpe_ratio, axis=0)


def rolling_sharpe(prices, window=250):
    """Rolling annualized Sharpe over a price Series."""
    rets = log_returns(prices)
    return rets.rolling(window).apply(sharpe_ratio, raw=False)


def plot_price_and_sharpe(prices, ticker=None, sharpe_window=250, ema_window=30):
    """
    Plot price on top and rolling Sharpe below.

    Pass a Series or a DataFrame plus ticker name.
    """
    import matplotlib.pyplot as plt

    if isinstance(prices, pd.DataFrame):
        if ticker is None:
            ticker = prices.columns[0]
        price = prices[ticker]
    else:
        price = prices
        ticker = price.name or "price"

    roll_sharpe = rolling_sharpe(price, window=sharpe_window)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(price, label=f"{ticker} price", linewidth=1.2)
    if ema_window:
        axes[0].plot(
            price.ewm(span=ema_window, adjust=False).mean(),
            label=f"EMA({ema_window})",
            linestyle="--",
        )
    axes[0].set_title(f"{ticker}: price")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(roll_sharpe, color="teal", label=f"Rolling Sharpe({sharpe_window})")
    axes[1].axhline(0.8, color="green", linestyle="--", alpha=0.7, label="long threshold")
    axes[1].axhline(-0.8, color="red", linestyle="--", alpha=0.7, label="short threshold")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Rolling Sharpe")
    axes[1].set_xlabel("Day")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return fig, axes
