"""
Keep the repo small as daily snapshots pile up.

Run from the project root:
    python -m src.prune

Raw snapshots in data/raw are the history, about 1 MB a day, and are never
touched. The cleaned and processed copies are only working files that every
run rebuilds, so for each ticker we keep the newest one and delete the rest.
Without this the repo would grow by around 15 MB a day.
"""

import logging
from pathlib import Path

TICKERS = ["SPY", "AAPL", "NVDA", "TSLA", "JPM"]
PATTERNS = [
    ("data/clean", "{t}_*_clean.parquet"),
    ("data/processed", "{t}_*_ready.parquet"),
    ("data/processed", "{t}_*_iv.parquet"),
    ("data/processed", "{t}_*_greeks.parquet"),
]

log = logging.getLogger("prune")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def main() -> None:
    removed = 0
    for folder, pattern in PATTERNS:
        for t in TICKERS:
            files = sorted(Path(folder).glob(pattern.format(t=t)))
            for old in files[:-1]:  # names carry a timestamp, so the last one is the newest
                old.unlink()
                removed += 1
            # Safety: the newest file must survive whatever happened above
            assert not files or files[-1].exists(), f"{files[-1]} went missing"
    log.info("removed %d old working files, raw snapshots untouched", removed)


if __name__ == "__main__":
    main()
