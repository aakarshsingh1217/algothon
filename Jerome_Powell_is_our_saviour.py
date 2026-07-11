"""Jerome Powell is our saviour — Algothon 2026 strategy.

Regime-based approach:

1. Classify each instrument by its recent Sharpe ratio.
   - High positive Sharpe   -> hold max long   (slow uptrend)
   - Very negative Sharpe   -> hold max short  (slow downtrend)
   - Everything else        -> mean-reversion using RSI + EMA + divergence
"""

import numpy as np
import pandas as pd

nInst = 51
currentPos = np.zeros(nInst, dtype=int)

# --- regime classification ---
SHARPE_LOOKBACK = 250
LONG_SHARPE_THRESHOLD = 0.8
SHORT_SHARPE_THRESHOLD = -0.8

# --- mean-reversion signals ---
RSI_WINDOW = 14
RSI_LOWER = 30
RSI_UPPER = 70
RSI_EXIT = 50

EMA_SLOW = 30

DIVERGENCE_LOOKBACK = 20

# --- position limits (matches eval.py) ---
DEFAULT_DLR_LIMIT = 10_000
INST0_DLR_LIMIT = 100_000


def _sharpe(prices):
    """Annualised Sharpe from a price series (log-return convention, sqrt(250))."""
    rets = np.log(prices / prices.shift(1)).dropna()
    if len(rets) < 2:
        return 0.0
    std = rets.std()
    if std < 1e-10:
        return 0.0
    return float(np.sqrt(250) * rets.mean() / std)


def _rsi(prices, window=RSI_WINDOW):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def _detect_divergence(prices, rsi_series, lookback=DIVERGENCE_LOOKBACK):
    """
    Simple divergence check over a rolling window.

    Returns:
      +1  bullish (price made a lower low, RSI made a higher low)
      -1  bearish (price made a higher high, RSI made a lower high)
       0  neither
    """
    if len(prices) < lookback or rsi_series.iloc[-lookback:].isna().any():
        return 0

    p = prices.iloc[-lookback:].reset_index(drop=True)
    r = rsi_series.iloc[-lookback:].reset_index(drop=True)
    mid = lookback // 2

    price_low_early, price_low_late = p.iloc[:mid].min(), p.iloc[mid:].min()
    rsi_low_early, rsi_low_late = r.iloc[:mid].min(), r.iloc[mid:].min()

    price_high_early, price_high_late = p.iloc[:mid].max(), p.iloc[mid:].max()
    rsi_high_early, rsi_high_late = r.iloc[:mid].max(), r.iloc[mid:].max()

    if price_low_late < price_low_early and rsi_low_late > rsi_low_early:
        return 1
    if price_high_late > price_high_early and rsi_high_late < rsi_high_early:
        return -1
    return 0


def getMyPosition(prcSoFar):
    """
    Called by eval.py each trading day.

    prcSoFar shape: (nInst, nDays), integer positions returned per instrument.
    """
    global currentPos
    nins, nt = prcSoFar.shape

    min_history = max(SHARPE_LOOKBACK, EMA_SLOW, RSI_WINDOW + 5, DIVERGENCE_LOOKBACK + 5)
    if nt < min_history:
        return np.zeros(nins, dtype=int)

    newPos = currentPos.copy()

    for i in range(nins):
        prices = pd.Series(prcSoFar[i, :])
        current_price = prices.iloc[-1]

        dlr_limit = INST0_DLR_LIMIT if i == 0 else DEFAULT_DLR_LIMIT
        max_shares = int(dlr_limit / current_price)

        sharpe = _sharpe(prices.iloc[-SHARPE_LOOKBACK:])

        # 1. Strong trend: hold max long / short.
        if sharpe >= LONG_SHARPE_THRESHOLD:
            newPos[i] = max_shares
            continue
        if sharpe <= SHORT_SHARPE_THRESHOLD:
            newPos[i] = -max_shares
            continue

        # 2. Sideways: mean-reversion with RSI, EMA filter, and divergence boost.
        rsi = _rsi(prices)
        ema_slow = prices.ewm(span=EMA_SLOW, adjust=False).mean()

        rsi_now = rsi.iloc[-1]
        rsi_prev = rsi.iloc[-2]
        div = _detect_divergence(prices, rsi)

        long_cross = rsi_prev < RSI_LOWER <= rsi_now
        short_cross = rsi_prev > RSI_UPPER >= rsi_now
        below_ema = current_price < ema_slow.iloc[-1]
        above_ema = current_price > ema_slow.iloc[-1]

        # Long: RSI crossed back above 30 AND price still under slow EMA (deep dip).
        # Bullish divergence alone can also trigger the entry.
        if (long_cross and below_ema) or div == 1:
            newPos[i] = max_shares
            continue

        # Short: RSI crossed back below 70 AND price above slow EMA. Same logic in reverse.
        if (short_cross and above_ema) or div == -1:
            newPos[i] = -max_shares
            continue

        # Exit when RSI passes back through the 50 midline against our position.
        pos = newPos[i]
        if pos > 0 and rsi_now >= RSI_EXIT:
            newPos[i] = 0
        elif pos < 0 and rsi_now <= RSI_EXIT:
            newPos[i] = 0

    currentPos = newPos.astype(int)
    return currentPos
