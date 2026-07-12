"""Exposure checking: measure the book against % -of-NAV policy limits.

Denominator is the cash-inclusive statement NAV (chosen over the intraday book,
which has no cash line). With no NAV the check refuses (MissingData) rather than
divide by a fabricated zero. First bucket implemented: options <= 10% NAV; the
"20% offensive exposure" bucket waits until 'offensive' is defined.
"""

from __future__ import annotations

from app.trading.domain import ExposureCheck, ExposureReport, MissingData, Position

_OPTIONS_LIMIT = 0.10  # options <= 10% of NAV 


def check_exposure(
    positions: list[Position], nav: float | None, options_limit: float = _OPTIONS_LIMIT
) -> ExposureReport | MissingData:
    if nav is None:
        return MissingData(
            store="statement NAV",
            remedy="Upload a recent activity statement (needed for the NAV denominator).",
        )

    options_value = sum(
        p.position_value * p.fx_rate_to_base
        for p in positions
        if p.asset_class == "OPT"
    )
    options = ExposureCheck(
        label="options",
        value_base=options_value,
        pct_of_nav=options_value / nav,
        limit=options_limit,
    )
    return ExposureReport(nav=nav, checks=(options,))
