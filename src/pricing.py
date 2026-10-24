"""
Step 4: three option pricers built from scratch. No pricing libraries.

    bs_price          Black-Scholes with a continuous dividend yield (European)
    binomial_price    Cox-Ross-Rubinstein tree, 500 steps, early exercise allowed
    mc_price          Monte Carlo, 100,000 paths with antithetic variates

Plus Greeks:
    bs_greeks         closed form
    binomial_greeks   finite differences on the tree

Units used everywhere in this file:
    S, K     dollars
    T        years
    r, q     continuously compounded annual rates, so 0.042 means 4.2%
    sigma    annual volatility, so 0.25 means 25%
    kind     "call" or "put"
"""

import numpy as np
from scipy.stats import norm


def _check_inputs(S, K, T, sigma, kind):
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    if np.any(np.asarray(S) <= 0) or np.any(np.asarray(K) <= 0):
        raise ValueError("spot and strike must be positive")
    if np.any(np.asarray(T) <= 0):
        raise ValueError("time to expiry must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("volatility must be positive")


def _d1_d2(S, K, T, r, q, sigma):
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------

def bs_price(S, K, T, r, q, sigma, kind="call"):
    """European option value. Works on single numbers or whole numpy arrays."""
    _check_inputs(S, K, T, sigma, kind)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_s = S * np.exp(-q * T)   # spot after stripping out expected dividends
    disc_k = K * np.exp(-r * T)   # strike in today's money
    if kind == "call":
        return disc_s * norm.cdf(d1) - disc_k * norm.cdf(d2)
    return disc_k * norm.cdf(-d2) - disc_s * norm.cdf(-d1)


def bs_greeks(S, K, T, r, q, sigma, kind="call"):
    """
    Closed form Greeks. Raw units, no rescaling:
        vega   price change for a 1.00 change in sigma (divide by 100 for per vol point)
        theta  price change per year (divide by 365 for per calendar day)
        rho    price change for a 1.00 change in r (divide by 100 for per 1%)
    """
    _check_inputs(S, K, T, sigma, kind)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    sqrt_t = np.sqrt(T)
    eq, er = np.exp(-q * T), np.exp(-r * T)
    pdf_d1 = norm.pdf(d1)

    # Gamma and vega are the same for calls and puts
    gamma = eq * pdf_d1 / (S * sigma * sqrt_t)
    vega = S * eq * pdf_d1 * sqrt_t
    decay = -S * eq * pdf_d1 * sigma / (2 * sqrt_t)

    if kind == "call":
        delta = eq * norm.cdf(d1)
        theta = decay - r * K * er * norm.cdf(d2) + q * S * eq * norm.cdf(d1)
        rho = K * T * er * norm.cdf(d2)
    else:
        delta = -eq * norm.cdf(-d1)
        theta = decay + r * K * er * norm.cdf(-d2) - q * S * eq * norm.cdf(-d1)
        rho = -K * T * er * norm.cdf(-d2)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


# ---------------------------------------------------------------------------
# Binomial tree
# ---------------------------------------------------------------------------

def binomial_price(S, K, T, r, q, sigma, kind="call", steps=500, american=True):
    """
    Cox-Ross-Rubinstein tree. With american=True every node checks whether
    exercising right now beats holding on, which is what makes US puts worth
    more than their European twins.

    Scalar inputs only. The backward pass is vectorised across each layer of
    the tree, so 500 steps takes a few milliseconds.
    """
    return _tree(S, K, T, r, q, sigma, kind, steps, american)["price"]


def _tree(S, K, T, r, q, sigma, kind, steps, american):
    """Runs the tree and also keeps the first two layers, which the Greeks need."""
    _check_inputs(S, K, T, sigma, kind)
    dt = T / steps
    up = np.exp(sigma * np.sqrt(dt))
    down = 1 / up
    p = (np.exp((r - q) * dt) - down) / (up - down)  # risk neutral chance of an up move
    if not 0 < p < 1:
        raise ValueError(f"tree is unstable (p={p:.4f}), try more steps")
    disc = np.exp(-r * dt)

    # Stock prices at expiry, from all downs (index 0) to all ups (index steps)
    j = np.arange(steps + 1)
    stock = S * up ** j * down ** (steps - j)
    payoff = np.maximum(stock - K, 0) if kind == "call" else np.maximum(K - stock, 0)
    values = payoff

    # Walk back one layer at a time, keeping layers 2 and 1 for the Greeks
    kept = {}
    for layer in range(steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1 - p) * values[:-1])
        if american:
            stock = stock[1:] * down  # step every node back one period in time
            exercise = np.maximum(stock - K, 0) if kind == "call" else np.maximum(K - stock, 0)
            values = np.maximum(values, exercise)
        if layer <= 2:
            kept[layer] = values.copy()

    return {"price": float(values[0]), "layers": kept, "up": up, "down": down, "dt": dt}


def binomial_greeks(S, K, T, r, q, sigma, kind="call", steps=500, american=True):
    """
    Greeks for the tree. Same raw units as bs_greeks.

    Delta, gamma and theta come straight from the first two layers of the tree
    (the textbook way, Hull chapter 21). Bumping the stock price and re-pricing
    looks simpler but gives noisy gamma, because each bump shifts where the
    strike falls between nodes.

    Vega and rho have no tree shortcut, so those use central finite differences:
    re-price with the input nudged up and down, and take the slope.
    """
    if steps < 3:
        raise ValueError("need at least 3 steps to read Greeks off the tree")
    t = _tree(S, K, T, r, q, sigma, kind, steps, american)
    u, d, dt = t["up"], t["down"], t["dt"]
    f1, f2 = t["layers"][1], t["layers"][2]  # index 0 is the lowest node

    # Layer 1 sits at stock prices S*d and S*u
    delta = (f1[1] - f1[0]) / (S * u - S * d)

    # Layer 2 sits at S*d^2, S, S*u^2. Gamma is the change in delta between them.
    delta_hi = (f2[2] - f2[1]) / (S * u * u - S)
    delta_lo = (f2[1] - f2[0]) / (S - S * d * d)
    gamma = (delta_hi - delta_lo) / (0.5 * (S * u * u - S * d * d))

    # The middle node of layer 2 has the same stock price as today, two steps later
    theta = (f2[1] - t["price"]) / (2 * dt)

    def price(rr=r, vol=sigma):
        return _tree(S, K, T, rr, q, vol, kind, steps, american)["price"]

    dv = 0.01    # 1 vol point
    dr = 0.0001  # 1 basis point
    return {
        "delta": delta,
        "gamma": gamma,
        "vega": (price(vol=sigma + dv) - price(vol=sigma - dv)) / (2 * dv),
        "theta": theta,
        "rho": (price(rr=r + dr) - price(rr=r - dr)) / (2 * dr),
    }


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def mc_price(S, K, T, r, q, sigma, kind="call", paths=100_000, seed=42):
    """
    European value by simulation. Antithetic variates: every random draw z is
    paired with -z, which cancels a big chunk of the noise for free.

    Returns a dict with the price, its standard error and a 95% confidence interval.
    The seed is fixed so results are reproducible run to run.
    """
    _check_inputs(S, K, T, sigma, kind)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(paths // 2)

    drift = (r - q - 0.5 * sigma ** 2) * T
    shock = sigma * np.sqrt(T)
    s_up = S * np.exp(drift + shock * z)
    s_dn = S * np.exp(drift - shock * z)

    if kind == "call":
        pay_up, pay_dn = np.maximum(s_up - K, 0), np.maximum(s_dn - K, 0)
    else:
        pay_up, pay_dn = np.maximum(K - s_up, 0), np.maximum(K - s_dn, 0)

    # Average each antithetic pair first. The pairs are independent of each
    # other, the two halves of a pair are not, so the error has to be measured
    # on the pair averages.
    pair = np.exp(-r * T) * 0.5 * (pay_up + pay_dn)
    price = pair.mean()
    std_err = pair.std(ddof=1) / np.sqrt(len(pair))
    return {
        "price": float(price),
        "std_err": float(std_err),
        "ci_low": float(price - 1.96 * std_err),
        "ci_high": float(price + 1.96 * std_err),
    }
