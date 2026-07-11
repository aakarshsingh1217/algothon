"""Cross-sectional short-term reversion strategy.

Every day:
  1. Compute the LOOKBACK-day log return per instrument.
  2. Cross-sectionally demean (dollar-neutral - long some, short others).
  3. Divide by rolling volatility (equal risk contribution).
  4. Flip sign to fade recent winners / buy recent losers.
  5. Scale to fixed dollar exposure per name, respecting the eval position limits.
"""

import numpy as np

nInst = 51
currentPos = np.zeros(nInst, dtype=int)

LOOKBACK = 6
VOL_WINDOW = 20
DOLLAR_PER_NAME = 8000
SMOOTH = 0.3

DEFAULT_DLR_LIMIT = 10_000
INST0_DLR_LIMIT = 100_000


def getMyPosition(prcSoFar):
    global currentPos
    nins, nt = prcSoFar.shape

    if nt < max(LOOKBACK, VOL_WINDOW) + 2:
        return np.zeros(nins, dtype=int)

    log_prices = np.log(prcSoFar)

    ret = log_prices[:, -1] - log_prices[:, -1 - LOOKBACK]
    ret -= ret.mean()

    daily_rets = np.diff(log_prices[:, -(VOL_WINDOW + 1):], axis=1)
    vol = daily_rets.std(axis=1)
    vol = np.where(vol < 1e-6, vol.mean() + 1e-6, vol)

    signal = -ret / vol
    scale = np.abs(signal).sum()
    if scale < 1e-9:
        return currentPos
    signal = signal / scale * nins

    current_prices = prcSoFar[:, -1]
    dlr_limit = np.full(nins, DEFAULT_DLR_LIMIT, dtype=float)
    dlr_limit[0] = INST0_DLR_LIMIT
    per_name_dollars = np.minimum(DOLLAR_PER_NAME, dlr_limit)
    per_name_dollars[0] = INST0_DLR_LIMIT

    target_shares = signal * per_name_dollars / current_prices

    max_shares = dlr_limit / current_prices
    target_shares = np.clip(target_shares, -max_shares, max_shares)

    smoothed = SMOOTH * currentPos + (1 - SMOOTH) * target_shares
    currentPos = smoothed.astype(int)
    return currentPos
