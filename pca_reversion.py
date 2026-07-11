"""PCA-residual cross-sectional mean reversion.

Strip out the top principal components (broad co-movement) from recent returns,
then fade the idiosyncratic residual. This isolates instrument-specific
reversion without being drowned out by market-wide moves.
"""

import numpy as np

nInst = 51
currentPos = np.zeros(nInst, dtype=int)

RET_WINDOW = 180          # days used to fit PCA
N_COMPONENTS = 7          # number of principal factors to strip
LOOKBACK = 1              # residual return lookback for the reversion signal
VOL_WINDOW = 40           # vol window used for equal-risk sizing
DOLLAR_PER_NAME = 9000    # target dollar exposure per instrument

DEFAULT_DLR_LIMIT = 10_000
INST0_DLR_LIMIT = 100_000


def getMyPosition(prcSoFar):
    global currentPos
    nins, nt = prcSoFar.shape

    if nt < max(RET_WINDOW, LOOKBACK, VOL_WINDOW) + 3:
        return np.zeros(nins, dtype=int)

    log_prices = np.log(prcSoFar)
    rets_window = np.diff(log_prices[:, -(RET_WINDOW + 1):], axis=1)

    # Strip top principal components (broad co-movement).
    mean_r = rets_window.mean(axis=1, keepdims=True)
    centered = rets_window - mean_r
    cov = centered @ centered.T / (RET_WINDOW - 1)
    _, eigvecs = np.linalg.eigh(cov)
    top = eigvecs[:, -N_COMPONENTS:]
    proj = top @ top.T
    residual = centered - proj @ centered

    resid_ret = residual[:, -LOOKBACK:].sum(axis=1)
    resid_ret -= resid_ret.mean()

    vol = residual[:, -VOL_WINDOW:].std(axis=1)
    vol = np.where(vol < 1e-6, vol.mean() + 1e-6, vol)

    signal = -resid_ret / vol
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

    currentPos = target_shares.astype(int)
    return currentPos
