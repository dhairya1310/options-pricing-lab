# Desk memo: what the October 23 close says about five option surfaces

**To:** Equity derivatives desk
**From:** Dhairya Shah
**Date:** October 24, 2026
**Data:** Yahoo Finance chains at the October 23, 2026 close, 8,568 contracts after cleaning. Rates: 3 month T-bill at 4.20%. Every figure below comes from `src/memo_stats.py` and can be regenerated.

## Bottom line

- **SPY carries by far the steepest skew:** 9.4 vol points between the 95% and 105% strikes at 30 days, about twice any of the four single names. TSLA is the flattest at 1.9.
- **Earnings are priced in visibly:** AAPL at ±4.3% for its November 2 report, NVDA at ±6.1% for November 18. Both show up as a bump in ATM vol on the first expiry after the report.
- **The options market funds about 65 bp over the T-bill,** read directly from the slope of call minus put across strikes. Pricing long dated options at the T-bill rate misprices them by dollars, not cents.

## The numbers

| | SPY | AAPL | NVDA | TSLA | JPM |
|---|---|---|---|---|---|
| ATM vol, 7 days | 16.2% | 22.7% | 37.8% | 39.4% | 21.9% |
| ATM vol, 30 days | 15.0% | 26.8% | 42.4% | 38.4% | 22.1% |
| ATM vol, 1 year | 17.3% | 27.0% | 39.4% | 43.3% | 24.9% |
| Term slope, 1 year minus 30 days | +2.3 | +0.2 | -3.0 | +4.9 | +2.8 |
| Skew, 30 days (95% minus 105%) | 9.4 | 4.9 | 4.0 | 1.9 | 4.2 |
| Skew, 90 days | 6.3 | 3.3 | 2.2 | 1.1 | 2.9 |
| Next earnings | none | Nov 2 | Nov 18 (expected) | Jan 27 | Jan 14 |
| Implied earnings move, 1 standard deviation | none | ±4.3% | ±6.1% | ±8.6% | ±3.4% |
| Funding over T-bill | 75 bp | 67 bp | 65 bp | 53 bp | 96 bp |

Vol figures are SVI ATM vols at the forward, interpolated in total variance. Skew and term slope are in vol points.

## Skew

The index carries the crash premium: portfolio hedgers buy index puts, so SPY's 9.4 points dwarfs the single names at 1.9 to 4.9. TSLA is the flattest, since demand for its upside calls offsets the put premium. Skew flattens with maturity on every name (SPY 9.4 to 6.3 points from 30 to 90 days). Ignore it and flat vol prices SPY puts struck below 90% of the forward with a median error of -100%: worthless to the model, real money to the market. On the single names that error runs from -21% (JPM) to -38% (NVDA).

## Term structure and events

- **SPY:** upward sloping overall, 15.0% at 30 days to 17.3% at 1 year, but the front week sits higher at 16.2%. The bump is on the October 30 expiry, the first after the October 28 FOMC decision.
- **AAPL:** 30 day vol (26.8%) is above both the 7 day (22.7%) and 90 day (24.9%) vols, because only the 30 day window contains the November 2 report. The options imply a ±4.3% move on the day. The bump sits on the November 6 expiry.
- **NVDA:** the only downward slope, -3.0 points. The November 18 report inflates the front: 30 day vol is 42.4% against 38.0% at 90 days. Implied move ±6.1%. The report date is the expected one, not yet confirmed.
- **TSLA:** steepest upward slope, +4.9 points. It also shows a bump on the November 6 expiry with no company event in that window. The November 3 midterm elections are the likely reason, but this data cannot prove it. Its ±8.6% January move is the roughest estimate here: it comes from the monthly expiries either side of the report, the later one (February 19) almost four months out.
- **JPM:** gently upward sloping, no visible event bump. It reports January 14 with an implied ±3.4% move.

**Implied move method:** jump variance = w_after - sigma_before² × T_after, using the expiries either side of the report. It assumes flat vol between them, so it is an estimate. On test data it recovered planted moves of 7.00% and 4.50% as 7.04% and 4.48%.

## Funding and put-call parity

1,134 of 2,471 call and put pairs break textbook parity by more than the bid/ask spread. The market's own forward explains 464 of them, American early exercise 624 more, and the funding rate 25 more. That leaves 21, and **none** break the no arbitrage range at bid and ask. Past one year, all five names imply a funding rate between 4.45% and 5.25% against the 4.20% T-bill, a median spread of 65 bp across 46 expiries.

## Caveats

A single snapshot of delayed end of day quotes. Midpoints can sit anywhere inside wide long dated spreads, dividends are a smooth yield, and JPM's thin chain makes its figures the least precise. The daily snapshot job will show how these surfaces move through the AAPL and NVDA reports.
