#!/usr/bin/env python3
"""Download 12 months of Uniswap metrics for Ethereum, Arbitrum, and Base.

Data sources
------------
* DefiLlama (no API key): daily TVL, DEX volume, and swap fees.
* Dune (optional API key): monthly end-of-month holder snapshots for UNI.

"Holder" means an address with a strictly positive balance of the official
bridged UNI ERC-20 on that chain. It does *not* mean a Uniswap LP: v3/v4 LP
positions are NFTs and require a separate position-holder query.

Examples
--------
python uniswap_data_pull.py
DUNE_API_KEY=... python uniswap_data_pull.py --with-dune-holders

Outputs are written to ./uniswap_output/.  No third-party Python packages are
required.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


LLAMA_BASE = "https://api.llama.fi"
DUNE_BASE = "https://api.dune.com/api/v1"
CHAINS = ("Ethereum", "Arbitrum", "Base")

# Official Uniswap developer documentation, checked September 2026.
UNI_TOKEN_BY_CHAIN = {
    "ethereum": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
    "arbitrum": "0xFa7F8980b0f1E64A2062791cc3b0871572f1F7f0",
    "base": "0xc3De830EA07524a0761646a6a4e4be0e114a3C83",
}


def get_json(url: str, headers: dict[str, str] | None = None, timeout: int = 60) -> Any:
    """GET JSON with a useful error message."""
    request = Request(url, headers=headers or {})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {exc.code} for {url}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc.reason}") from exc


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int = 60) -> Any:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {exc.code} for {url}: {detail}") from exc


def utc_day(timestamp: int | float) -> str:
    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).date().isoformat()


def cutoff_timestamp(months: int) -> int:
    # 366 days covers a 12-month calendar period without external dependencies.
    return int((datetime.now(timezone.utc) - timedelta(days=366 if months == 12 else months * 31)).timestamp())


def chain_key(record: dict[str, Any], chain: str) -> str | None:
    """Find a chain in a DefiLlama breakdown, allowing API capitalization changes."""
    target = chain.lower()
    for key in record:
        if key.lower() == target:
            return key
    return None


def unpack_breakdown(payload: dict[str, Any], metric: str, cutoff: int) -> list[dict[str, Any]]:
    """Convert DefiLlama's [timestamp, {chain: value}] chart into tidy rows.

    The public summary endpoints currently provide ``totalDataChartBreakdown``.
    Some historic responses use an object keyed by timestamp, so both shapes are
    handled here.  An explicit failure is safer than silently returning an
    all-chain total when a chain breakdown is unavailable.
    """
    chart = payload.get("totalDataChartBreakdown")
    if not chart:
        available = ", ".join(payload.keys())
        raise RuntimeError(
            f"DefiLlama did not return totalDataChartBreakdown for {metric}. "
            f"Available fields: {available}"
        )

    rows: list[dict[str, Any]] = []
    if isinstance(chart, dict):
        points: Iterable[tuple[Any, Any]] = chart.items()
    else:
        points = ((point[0], point[1]) for point in chart if len(point) >= 2)

    for raw_timestamp, breakdown in points:
        timestamp = int(float(raw_timestamp))
        if timestamp < cutoff or not isinstance(breakdown, dict):
            continue
        for chain in CHAINS:
            source_key = chain_key(breakdown, chain)
            if source_key is not None:
                value = breakdown[source_key]
                if isinstance(value, dict):
                    metric_value = sum(
                        float(component) for component in value.values() if component is not None
                    )
                else:
                    metric_value = float(value) if value is not None else None
                rows.append(
                    {
                        "date": utc_day(timestamp),
                        "chain": chain.lower(),
                        metric: metric_value,
                    }
                )
    return rows


def fetch_defillama(months: int) -> list[dict[str, Any]]:
    """Fetch and merge daily chain-level TVL, volume, and fees."""
    cutoff = cutoff_timestamp(months)
    protocol = get_json(f"{LLAMA_BASE}/protocol/uniswap-v3")
    chain_tvls = protocol.get("chainTvls", {})

    tvl_rows: list[dict[str, Any]] = []
    for chain in CHAINS:
        chain_data = chain_tvls.get(chain) or chain_tvls.get(chain.lower())
        chart = chain_data.get("tvl") if isinstance(chain_data, dict) else chain_data
        if not isinstance(chart, list):
            raise RuntimeError(f"Uniswap protocol payload has no TVL chart for {chain}.")
        for point in chart:
            timestamp = int(point["date"])
            if timestamp >= cutoff:
                tvl_rows.append(
                    {
                        "date": utc_day(timestamp),
                        "chain": chain.lower(),
                        "tvl_usd": point.get("totalLiquidityUSD"),
                    }
                )

    # These are Uniswap's combined protocol summaries. dataType requests daily,
    # rather than rolling 7d/30d, values.
    volume_url = f"{LLAMA_BASE}/summary/dexs/uniswap?" + urlencode({"dataType": "dailyVolume"})
    fees_url = f"{LLAMA_BASE}/summary/fees/uniswap?" + urlencode({"dataType": "dailyFees"})
    volume_rows = unpack_breakdown(get_json(volume_url), "volume_usd", cutoff)
    fee_rows = unpack_breakdown(get_json(fees_url), "fees_usd", cutoff)

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for rows in (tvl_rows, volume_rows, fee_rows):
        for row in rows:
            key = (row["date"], row["chain"])
            merged.setdefault(key, {"date": row["date"], "chain": row["chain"]}).update(row)

    return [
        {
            "date": row["date"],
            "chain": row["chain"],
            "tvl_usd": row.get("tvl_usd"),
            "volume_usd": row.get("volume_usd"),
            "fees_usd": row.get("fees_usd"),
        }
        for _, row in sorted(merged.items())
    ]


# The EVM balance schemas are gated by Dune. This is the fast and accurate
# method because Dune reads balances from chain state rather than attempting to
# rebuild them from every Transfer event. The query returns 12 month-end
# snapshots of count of holders and aggregate UNI balance, by chain.
DUNE_HOLDERS_SQL = r"""
WITH token_config (chain, token_address) AS (
    VALUES
        ('ethereum', 0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984),
        ('arbitrum', 0xFa7F8980b0f1E64A2062791cc3b0871572f1F7f0),
        ('base', 0xc3De830EA07524a0761646a6a4e4be0e114a3C83)
),
month_ends AS (
    SELECT date_add('day', -1, date_add('month', 1, month_start)) AS snapshot_date
    FROM UNNEST(sequence(
        date_trunc('month', current_date - interval '11' month),
        date_trunc('month', current_date),
        interval '1' month
    )) AS t(month_start)
),
ethereum AS (
    SELECT m.snapshot_date, 'ethereum' AS chain, b.address, b.balance
    FROM month_ends m
    JOIN balances_ethereum.daily_updates b
      ON m.snapshot_date >= b.valid_from
     AND m.snapshot_date < b.valid_to
    WHERE b.token_address = (SELECT token_address FROM token_config WHERE chain = 'ethereum')
      AND b.valid_from <= (SELECT max(snapshot_date) FROM month_ends)
      AND b.valid_to > (SELECT min(snapshot_date) FROM month_ends)
),
arbitrum AS (
    SELECT m.snapshot_date, 'arbitrum' AS chain, b.address, b.balance
    FROM month_ends m
    JOIN balances_arbitrum.daily_updates b
      ON m.snapshot_date >= b.valid_from
     AND m.snapshot_date < b.valid_to
    WHERE b.token_address = (SELECT token_address FROM token_config WHERE chain = 'arbitrum')
      AND b.valid_from <= (SELECT max(snapshot_date) FROM month_ends)
      AND b.valid_to > (SELECT min(snapshot_date) FROM month_ends)
),
base AS (
    SELECT m.snapshot_date, 'base' AS chain, b.address, b.balance
    FROM month_ends m
    JOIN balances_base.daily_updates b
      ON m.snapshot_date >= b.valid_from
     AND m.snapshot_date < b.valid_to
    WHERE b.token_address = (SELECT token_address FROM token_config WHERE chain = 'base')
      AND b.valid_from <= (SELECT max(snapshot_date) FROM month_ends)
      AND b.valid_to > (SELECT min(snapshot_date) FROM month_ends)
)
SELECT
    snapshot_date,
    chain,
    COUNT(*) AS holder_count,
    SUM(balance) AS uni_balance
FROM (
    SELECT * FROM ethereum
    UNION ALL SELECT * FROM arbitrum
    UNION ALL SELECT * FROM base
) balances
WHERE balance > 0
  -- Exclude the ERC-20 zero address (mint/burn accounting), not real holders.
  AND address <> 0x0000000000000000000000000000000000000000
GROUP BY 1, 2
ORDER BY 1, 2
""".strip()


def execute_dune_sql(sql: str, api_key: str) -> list[dict[str, Any]]:
    """Run raw DuneSQL, wait for completion, then return all result rows."""
    headers = {"X-Dune-API-Key": api_key}
    start = post_json(
        f"{DUNE_BASE}/sql/execute",
        {"sql": sql, "performance": "medium"},
        headers,
    )
    execution_id = start["execution_id"]
    deadline = time.monotonic() + 15 * 60
    while time.monotonic() < deadline:
        status = get_json(f"{DUNE_BASE}/execution/{execution_id}/status", headers=headers)
        state = status.get("state")
        if state == "QUERY_STATE_COMPLETED":
            break
        if state in {"QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"}:
            raise RuntimeError(f"Dune query {execution_id} ended in {state}: {status}")
        time.sleep(2)
    else:
        raise TimeoutError(f"Dune query {execution_id} did not finish within 15 minutes.")

    result = get_json(f"{DUNE_BASE}/execution/{execution_id}/results", headers=headers)
    return result.get("result", {}).get("rows", [])


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", type=int, default=12, help="History length (default: 12).")
    parser.add_argument("--output-dir", default="uniswap_output", help="Directory for CSV output.")
    parser.add_argument(
        "--with-dune-holders",
        action="store_true",
        help="Run the optional Dune UNI-holder query; requires DUNE_API_KEY and access to Dune EVM balances.",
    )
    args = parser.parse_args()
    if args.months < 1:
        parser.error("--months must be at least 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = fetch_defillama(args.months)
    metrics_path = output_dir / "uniswap_daily_metrics.csv"
    write_csv(metrics_path, metrics, ["date", "chain", "tvl_usd", "volume_usd", "fees_usd"])
    print(f"Wrote {len(metrics):,} daily metric rows to {metrics_path}")

    if args.with_dune_holders:
        key = os.environ.get("DUNE_API_KEY")
        if not key:
            parser.error("Set DUNE_API_KEY before using --with-dune-holders.")
        # The stored SQL is also a copy/paste-ready 12-month query. Adjust it
        # here if callers request another history length.
        holder_sql = DUNE_HOLDERS_SQL.replace("interval '11' month", f"interval '{args.months - 1}' month")
        holders = execute_dune_sql(holder_sql, key)
        holders_path = output_dir / "uniswap_uni_holders_month_end.csv"
        write_csv(holders_path, holders, ["snapshot_date", "chain", "holder_count", "uni_balance"])
        print(f"Wrote {len(holders):,} holder snapshot rows to {holders_path}")


if __name__ == "__main__":
    main()
