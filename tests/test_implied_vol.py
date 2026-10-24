"""
Tests for src/implied_vol.py. Run from the project root with:
    python -m pytest -v
"""

import numpy as np
import pytest

from src.implied_vol import implied_vol
from src.pricing import binomial_price, bs_price

S, T, R, Q = 100.0, 0.5, 0.042, 0.01


@pytest.mark.parametrize("sigma", [0.05, 0.15, 0.30, 0.80, 2.00])
@pytest.mark.parametrize("K", [70, 100, 130])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_round_trip_recovers_the_vol(sigma, K, kind):
    # Price with a known vol, then ask the solver to find it again
    price = bs_price(S, K, T, R, Q, sigma, kind)
    iv, status = implied_vol(price, S, K, T, R, Q, kind)

    # Two of these combos are deep in the money at 5% vol, where the price
    # does not move with vol at all. Those must be flagged, everything else solved.
    one_point = bs_price(S, K, T, R, Q, sigma + 0.01, kind) - price
    if abs(one_point) < 0.001:
        assert status == "vega too small" and np.isnan(iv)
    else:
        assert status == "ok"
        assert iv == pytest.approx(sigma, abs=1e-6)


def test_price_below_intrinsic_has_no_solution():
    # A call on a 100 stock with strike 80 cannot be worth 5
    iv, status = implied_vol(5.0, S, 80, T, R, Q, "call")
    assert np.isnan(iv) and status == "below model minimum"


def test_price_above_stock_has_no_solution():
    # No call can cost more than the stock itself
    iv, status = implied_vol(120.0, S, 100, T, R, Q, "call")
    assert np.isnan(iv) and status == "above model maximum"


def test_deep_itm_american_put_is_flagged_not_guessed():
    # A deep ITM American put trades above what any European put can be worth
    # at low vol. The solver must either flag it or return a vol above the true one,
    # never a silently wrong low number.
    true_vol = 0.25
    american = binomial_price(S, 150, 1.0, 0.05, 0.0, true_vol, "put")
    iv, status = implied_vol(american, S, 150, 1.0, 0.05, 0.0, "put")
    assert status == "below model minimum" or iv > true_vol


def test_otm_american_put_iv_is_close_to_true_vol():
    # Out of the money, early exercise is worth very little, so BS IV on an
    # American price should land within half a vol point of the truth
    true_vol = 0.25
    american = binomial_price(S, 85, 0.25, R, Q, true_vol, "put")
    iv, status = implied_vol(american, S, 85, 0.25, R, Q, "put")
    assert status == "ok"
    assert iv == pytest.approx(true_vol, abs=0.005)
