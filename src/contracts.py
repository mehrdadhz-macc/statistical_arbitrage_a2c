"""
List delivery-hour contracts (one A2C episode each) for a data split, built
on top of src.data_loader.build_day_index (kept verbatim/unmodified from
reinforce_threshold_policy -- it already does exactly the day/hour bookkeeping
we need, just grouped by day instead of flattened).
"""

from __future__ import annotations

import pandas as pd

from src.data_loader import build_day_index


def list_contracts(
    cim: pd.DataFrame,
    auc: pd.DataFrame,
    days: int | None = None,
) -> list[pd.Timestamp]:
    """Chronological list of delivery_start timestamps across the first `days` days."""
    day_index = build_day_index(cim, auc)
    if days is not None:
        day_index = day_index[:days]
    contracts: list[pd.Timestamp] = []
    for _, delivery_starts, _ in day_index:
        contracts.extend(delivery_starts)
    return contracts
