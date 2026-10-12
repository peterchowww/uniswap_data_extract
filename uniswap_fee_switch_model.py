#!/usr/bin/env python3
"""Model a hypothetical Uniswap protocol fee switch and UNI valuation.

The input is the `uniswap_daily_metrics.csv` file created by
`uniswap_data_pull.py`.  It must contain at least: date, chain, volume_usd.
The model sums the observed Ethereum, Arbitrum, and Base volume, annualizes it
to 365 days, and applies protocol take-rates of 5, 10, and 20 basis points.

This is an illustrative *protocol-fee* model. It does not assume those bps are
the underlying liquidity-provider swap fee tiers. A 100% UNI capture rate means
all modeled protocol fee revenue is used for UNI buybacks/burns or otherwise
accrues to UNI holders; change --uni-capture-rate if that is not your view.

Examples
--------
python uniswap_fee_switch_model.py
python uniswap_fee_switch_model.py --growth-rate 0.20 --discount-rate 0.14
python uniswap_fee_switch_model.py --token-supply 850000000 --uni-capture-rate 0.75
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any


DEFAULT_VOLUME_FILE = Path("uniswap_output/uniswap_daily_metrics.csv")
DEFAULT_OUTPUT_DIR = Path("uniswap_output")
FEE_TIERS_BPS = (5, 10, 20)
P_FEES_MULTIPLES = (10, 15, 20, 25, 30)
DAYS_IN_YEAR = 365


@dataclass(frozen=True)
class VolumeHistory:
    start_date: str
    end_date: str
    calendar_days: int
    observed_volume_usd: float
    annualized_volume_usd: float


@dataclass(frozen=True)
class FeeScenario:
    fee_bps: int
    protocol_take_rate: float
    annualized_volume_usd: float
    annual_protocol_revenue_usd: float
    uni_capture_rate: float
    annual_uni_value_accrual_usd: float


def to_float(value: str | None, field: str) -> float:
    try:
        return float(value) if value not in (None, "") else 0.0
    except ValueError as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc


def load_and_annualize_volume(path: Path) -> VolumeHistory:
    """Read daily chain volume rows and annualize their combined dollar volume."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Volume input not found: {path}. Run uniswap_data_pull.py first, "
            "or pass --volume-file PATH."
        )

    daily_volume: dict[date, float] = defaultdict(float)
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"date", "chain", "volume_usd"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must have columns: {', '.join(sorted(required))}")
        for row in reader:
            if not row.get("volume_usd"):
                continue
            try:
                row_date = date.fromisoformat(row["date"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid ISO date in input: {row.get('date')!r}") from exc
            daily_volume[row_date] += to_float(row["volume_usd"], "volume_usd")

    if not daily_volume:
        raise ValueError(f"No numeric volume_usd rows found in {path}.")

    start, end = min(daily_volume), max(daily_volume)
    calendar_days = (end - start).days + 1
    observed_volume = sum(daily_volume.values())
    return VolumeHistory(
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        calendar_days=calendar_days,
        observed_volume_usd=observed_volume,
        annualized_volume_usd=observed_volume * DAYS_IN_YEAR / calendar_days,
    )


def build_fee_scenarios(history: VolumeHistory, fee_tiers_bps: tuple[int, ...], capture_rate: float) -> list[FeeScenario]:
    scenarios = []
    for bps in fee_tiers_bps:
        take_rate = bps / 10_000
        annual_revenue = history.annualized_volume_usd * take_rate
        scenarios.append(
            FeeScenario(
                fee_bps=bps,
                protocol_take_rate=take_rate,
                annualized_volume_usd=history.annualized_volume_usd,
                annual_protocol_revenue_usd=annual_revenue,
                uni_capture_rate=capture_rate,
                annual_uni_value_accrual_usd=annual_revenue * capture_rate,
            )
        )
    return scenarios


def p_fees_value(annual_uni_value_accrual: float, multiple: float, token_supply: float) -> tuple[float, float]:
    """Return implied UNI equity value and price for a P/Fees-like multiple."""
    equity_value = annual_uni_value_accrual * multiple
    return equity_value, equity_value / token_supply


def dcf_value(
    base_year_uni_value_accrual: float,
    growth_rate: float,
    discount_rate: float,
    terminal_growth_rate: float,
    token_supply: float,
    years: int = 5,
) -> tuple[float, float]:
    """Two-stage DCF: five explicit annual cash flows plus Gordon-growth terminal value."""
    if discount_rate <= terminal_growth_rate:
        raise ValueError("discount_rate must be greater than terminal_growth_rate for a valid Gordon-growth DCF.")
    if years < 1:
        raise ValueError("years must be at least 1.")

    present_value = 0.0
    year_fcf = base_year_uni_value_accrual
    for year in range(1, years + 1):
        year_fcf *= 1 + growth_rate
        present_value += year_fcf / (1 + discount_rate) ** year

    terminal_value = year_fcf * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
    present_value += terminal_value / (1 + discount_rate) ** years
    return present_value, present_value / token_supply


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume-file", type=Path, default=DEFAULT_VOLUME_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--token-supply",
        type=float,
        default=1_000_000_000,
        help="UNI tokens used to calculate per-token values (default: 1,000,000,000).",
    )
    parser.add_argument(
        "--uni-capture-rate",
        type=float,
        default=1.0,
        help="Share of protocol-fee revenue assumed to accrue to UNI (default: 1.0).",
    )
    parser.add_argument("--growth-rate", type=float, default=0.15, help="Annual UNI-accrual growth for years 1-5 (default: 0.15).")
    parser.add_argument("--discount-rate", type=float, default=0.12, help="DCF discount rate (default: 0.12).")
    parser.add_argument("--terminal-growth-rate", type=float, default=0.03, help="DCF terminal growth rate (default: 0.03).")
    args = parser.parse_args()

    if args.token_supply <= 0:
        parser.error("--token-supply must be positive.")
    if not 0 <= args.uni_capture_rate <= 1:
        parser.error("--uni-capture-rate must be between 0 and 1.")
    if args.discount_rate <= args.terminal_growth_rate:
        parser.error("--discount-rate must be greater than --terminal-growth-rate.")

    history = load_and_annualize_volume(args.volume_file)
    scenarios = build_fee_scenarios(history, FEE_TIERS_BPS, args.uni_capture_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    revenue_rows = [asdict(scenario) for scenario in scenarios]
    write_csv(
        args.output_dir / "uniswap_fee_switch_revenue.csv",
        revenue_rows,
        list(revenue_rows[0]),
    )

    valuation_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for multiple in P_FEES_MULTIPLES:
            equity_value, price = p_fees_value(
                scenario.annual_uni_value_accrual_usd, multiple, args.token_supply
            )
            valuation_rows.append(
                {
                    "method": "P/Fees",
                    "fee_bps": scenario.fee_bps,
                    "multiple": multiple,
                    "implied_equity_value_usd": equity_value,
                    "implied_uni_price_usd": price,
                }
            )

        equity_value, price = dcf_value(
            scenario.annual_uni_value_accrual_usd,
            args.growth_rate,
            args.discount_rate,
            args.terminal_growth_rate,
            args.token_supply,
        )
        valuation_rows.append(
            {
                "method": "DCF (5-year + terminal)",
                "fee_bps": scenario.fee_bps,
                "multiple": None,
                "implied_equity_value_usd": equity_value,
                "implied_uni_price_usd": price,
            }
        )

    write_csv(
        args.output_dir / "uni_valuation_sensitivity.csv",
        valuation_rows,
        ["method", "fee_bps", "multiple", "implied_equity_value_usd", "implied_uni_price_usd"],
    )

    summary = {
        "input": str(args.volume_file),
        "volume_history": asdict(history),
        "assumptions": {
            "fee_tiers_bps": list(FEE_TIERS_BPS),
            "p_fees_multiples": list(P_FEES_MULTIPLES),
            "token_supply": args.token_supply,
            "uni_capture_rate": args.uni_capture_rate,
            "five_year_growth_rate": args.growth_rate,
            "discount_rate": args.discount_rate,
            "terminal_growth_rate": args.terminal_growth_rate,
        },
        "fee_scenarios": revenue_rows,
    }
    with (args.output_dir / "uniswap_fee_switch_summary.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)

    print(f"Observed volume ({history.start_date} to {history.end_date}): ${history.observed_volume_usd:,.0f}")
    print(f"Annualized volume: ${history.annualized_volume_usd:,.0f}")
    for scenario in scenarios:
        print(
            f"{scenario.fee_bps:>2} bps -> annual protocol revenue "
            f"${scenario.annual_protocol_revenue_usd:,.0f}; UNI-accrual "
            f"${scenario.annual_uni_value_accrual_usd:,.0f}"
        )
    print(f"Wrote model outputs to {args.output_dir}/")


if __name__ == "__main__":
    main()
