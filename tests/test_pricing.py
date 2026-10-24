"""
Tests for src/pricing.py. Run from the project root with:
    python -m pytest -v

Every test checks a pricer against something we know has to be true:
a textbook answer, an arbitrage bound, or another pricer.
"""

import numpy as np
import pytest

from src.pricing import (
    binomial_greeks,
    binomial_price,
    bs_greeks,
    bs_price,
    mc_price,
)

# A realistic contract close to the SPY data: one year, 4.2% rate, 1% dividend
BASE = dict(S=100.0, K=100.0, T=1.0, r=0.042, q=0.01, sigma=0.20)


# ---------------------------------------------------------------------------
# Black-Scholes against known answers and arbitrage rules
# ---------------------------------------------------------------------------

def test_bs_matches_hull_textbook():
    # Hull, Options Futures and Other Derivatives, Example 15.6:
    # S=42, K=40, r=10%, sigma=20%, T=0.5, no dividend. Call 4.76, put 0.81
    assert bs_price(42, 40, 0.5, 0.10, 0.0, 0.20, "call") == pytest.approx(4.76, abs=0.005)
    assert bs_price(42, 40, 0.5, 0.10, 0.0, 0.20, "put") == pytest.approx(0.81, abs=0.005)


@pytest.mark.parametrize("K", [60, 90, 100, 110, 150])
def test_bs_put_call_parity(K):
    # C - P = S e^(-qT) - K e^(-rT), has to hold exactly for European options
    b = {**BASE, "K": K}
    lhs = bs_price(**b, kind="call") - bs_price(**b, kind="put")
    rhs = b["S"] * np.exp(-b["q"] * b["T"]) - K * np.exp(-b["r"] * b["T"])
    assert lhs == pytest.approx(rhs, abs=1e-10)


@pytest.mark.parametrize("K", [50, 100, 200])
def test_bs_stays_inside_no_arbitrage_bounds(K):
    b = {**BASE, "K": K}
    fwd_s = b["S"] * np.exp(-b["q"] * b["T"])
    pv_k = K * np.exp(-b["r"] * b["T"])
    call = bs_price(**b, kind="call")
    put = bs_price(**b, kind="put")
    assert max(fwd_s - pv_k, 0) <= call <= fwd_s
    assert max(pv_k - fwd_s, 0) <= put <= pv_k


def test_bs_works_on_arrays():
    strikes = np.array([90.0, 100.0, 110.0])
    prices = bs_price(100.0, strikes, 1.0, 0.042, 0.01, 0.2, "call")
    assert prices.shape == (3,)
    assert np.all(np.diff(prices) < 0)  # higher strike, cheaper call


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        bs_price(100, 100, 0, 0.04, 0, 0.2)
    with pytest.raises(ValueError):
        bs_price(100, 100, 1, 0.04, 0, -0.2)
    with pytest.raises(ValueError):
        bs_price(100, 100, 1, 0.04, 0, 0.2, "straddle")


# ---------------------------------------------------------------------------
# Binomial tree
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("K", [80, 100, 120])
def test_european_binomial_converges_to_bs(kind, K):
    b = {**BASE, "K": K}
    tree = binomial_price(**b, kind=kind, steps=500, american=False)
    assert tree == pytest.approx(bs_price(**b, kind=kind), abs=0.01)


@pytest.mark.parametrize("K", [80, 100, 120])
def test_american_put_worth_at_least_european(K):
    b = {**BASE, "K": K}
    american = binomial_price(**b, kind="put", american=True)
    european = binomial_price(**b, kind="put", american=False)
    assert american >= european - 1e-12
    assert american >= K - b["S"]  # never below what you get by exercising today


def test_deep_itm_american_put_has_early_exercise_premium():
    # With rates at 4% and a strike far above spot, waiting costs interest,
    # so early exercise must add real value
    b = {**BASE, "K": 150, "q": 0.0}
    gap = binomial_price(**b, kind="put", american=True) - bs_price(**b, kind="put")
    assert gap > 0.5


def test_american_call_without_dividend_never_exercised_early():
    # Classic result: with no dividend, an American call equals the European one
    b = {**BASE, "q": 0.0}
    american = binomial_price(**b, kind="call", american=True)
    assert american == pytest.approx(bs_price(**b, kind="call"), abs=0.01)


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("K", [80, 100, 120])
def test_mc_confidence_interval_contains_bs(kind, K):
    b = {**BASE, "K": K}
    mc = mc_price(**b, kind=kind)
    exact = bs_price(**b, kind=kind)
    # A 99.7% band, so the test does not fail on ordinary bad luck
    assert abs(mc["price"] - exact) < 3 * mc["std_err"]
    assert mc["ci_low"] < mc["price"] < mc["ci_high"]


def test_mc_is_reproducible():
    assert mc_price(**BASE)["price"] == mc_price(**BASE)["price"]


def test_antithetic_actually_reduces_error():
    # Same number of draws without the antithetic pairing should be noisier
    b = BASE
    rng = np.random.default_rng(42)
    z = rng.standard_normal(100_000)
    s_t = b["S"] * np.exp((b["r"] - b["q"] - 0.5 * b["sigma"] ** 2) * b["T"]
                          + b["sigma"] * np.sqrt(b["T"]) * z)
    plain = np.exp(-b["r"] * b["T"]) * np.maximum(s_t - b["K"], 0)
    plain_err = plain.std(ddof=1) / np.sqrt(len(plain))
    assert mc_price(**b)["std_err"] < plain_err


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["call", "put"])
def test_bs_greeks_match_numerical_bumps(kind):
    # Re-derive each closed form Greek by bumping bs_price directly
    b = BASE
    g = bs_greeks(**b, kind=kind)
    h = 1e-4

    def p(**over):
        return bs_price(**{**b, **over}, kind=kind)

    assert g["delta"] == pytest.approx((p(S=b["S"] + h) - p(S=b["S"] - h)) / (2 * h), rel=1e-5)
    assert g["gamma"] == pytest.approx((p(S=b["S"] + h) - 2 * p() + p(S=b["S"] - h)) / h ** 2, rel=1e-3)
    assert g["vega"] == pytest.approx((p(sigma=b["sigma"] + h) - p(sigma=b["sigma"] - h)) / (2 * h), rel=1e-5)
    assert g["rho"] == pytest.approx((p(r=b["r"] + h) - p(r=b["r"] - h)) / (2 * h), rel=1e-5)
    # Theta is minus the sensitivity to T
    assert g["theta"] == pytest.approx(-(p(T=b["T"] + h) - p(T=b["T"] - h)) / (2 * h), rel=1e-5)


def test_bs_greek_signs_make_sense():
    c = bs_greeks(**BASE, kind="call")
    p = bs_greeks(**BASE, kind="put")
    assert 0 < c["delta"] < 1 and -1 < p["delta"] < 0
    assert c["gamma"] > 0 and c["vega"] > 0
    assert c["gamma"] == pytest.approx(p["gamma"]) and c["vega"] == pytest.approx(p["vega"])
    assert c["rho"] > 0 > p["rho"]
    # Call delta minus put delta equals e^(-qT)
    assert c["delta"] - p["delta"] == pytest.approx(np.exp(-BASE["q"] * BASE["T"]))


@pytest.mark.parametrize("kind", ["call", "put"])
def test_european_tree_greeks_close_to_bs(kind):
    bs = bs_greeks(**BASE, kind=kind)
    tree = binomial_greeks(**BASE, kind=kind, american=False)
    assert tree["delta"] == pytest.approx(bs["delta"], abs=0.005)
    assert tree["gamma"] == pytest.approx(bs["gamma"], rel=0.05)
    assert tree["vega"] == pytest.approx(bs["vega"], rel=0.02)
    assert tree["theta"] == pytest.approx(bs["theta"], rel=0.05)
    assert tree["rho"] == pytest.approx(bs["rho"], rel=0.02)
