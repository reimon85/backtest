"""EEVS audit — CSV audit trail with SHA-256 verification.

Writes deterministic CSVs with metadata headers for cross-tool comparison.
"""

import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

from .config import StrategyConfig
from .simulator import TradeRecord


def _config_hash(config: StrategyConfig) -> str:
    """Deterministic hash of strategy configuration."""
    d = {
        "name": config.name,
        "version": config.version,
        "asset": config.asset,
        "timeframe": config.timeframe,
        "params": config.params,
        "warmup_bars": config.warmup_bars,
        "tp_points": config.tp_points,
        "sl_points": config.sl_points,
        "entry_mode": config.entry_mode,
        "worst_case_intrabar": config.worst_case_intrabar,
        "cost_round_trip": config.cost_model.round_trip,
    }
    raw = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _file_hash(path: Path) -> str:
    """SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _data_hash(config: StrategyConfig) -> str:
    """SHA-256 of the source data file (first 16 chars)."""
    return _file_hash(Path(config.data_path))[:16]


TRADE_COLUMNS = [
    "signal_time", "entry_time", "direction", "signal_price", "fill_price",
    "tp_level", "sl_level", "exit_time", "exit_price", "exit_reason",
    "pnl_gross", "pnl_net", "cost_applied", "month", "pnl_eq",
]


def write_audit_csv(
    trades: list[TradeRecord],
    config: StrategyConfig,
    output_dir: str = "eevs_output",
) -> tuple[Path, str]:
    """Write audit CSV with metadata header. Returns (path, sha256)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{config.name}_{config.version}_{timestamp}.csv"
    filepath = out / filename

    with open(filepath, "w", newline="") as f:
        # Metadata header (commented lines)
        f.write(f"# EEVS Audit CSV\n")
        f.write(f"# strategy: {config.name} v{config.version}\n")
        f.write(f"# asset: {config.asset}\n")
        f.write(f"# timeframe: {config.timeframe}\n")
        f.write(f"# config_hash: {_config_hash(config)}\n")
        f.write(f"# data_hash: {_data_hash(config)}\n")
        f.write(f"# timestamp: {timestamp}\n")
        f.write(f"# entry_mode: {config.entry_mode}\n")
        f.write(f"# cost_round_trip: {config.cost_model.round_trip}\n")
        f.write(f"# total_trades: {len(trades)}\n")
        f.write(f"#\n")

        writer = csv.writer(f)
        writer.writerow(TRADE_COLUMNS)
        for t in trades:
            writer.writerow([
                t.signal_time, t.entry_time, t.direction, t.signal_price,
                t.fill_price, t.tp_level, t.sl_level, t.exit_time,
                t.exit_price, t.exit_reason, t.pnl_gross, t.pnl_net,
                t.cost_applied, t.month, t.pnl_eq,
            ])

    sha = _file_hash(filepath)
    return filepath, sha


def verify_audit(file_a: str, file_b: str) -> dict:
    """Compare two audit CSVs trade by trade.

    Returns dict with match status and list of mismatches.
    """
    def _read_trades(path: str) -> list[dict]:
        rows = []
        with open(path) as f:
            lines = [l for l in f if not l.startswith("#")]
        reader = csv.DictReader(lines)
        for row in reader:
            rows.append(row)
        return rows

    trades_a = _read_trades(file_a)
    trades_b = _read_trades(file_b)

    result = {
        "file_a": file_a,
        "file_b": file_b,
        "trades_a": len(trades_a),
        "trades_b": len(trades_b),
        "match": True,
        "mismatches": [],
    }

    if len(trades_a) != len(trades_b):
        result["match"] = False
        result["mismatches"].append(
            f"Trade count differs: {len(trades_a)} vs {len(trades_b)}"
        )
        return result

    compare_fields = [
        "signal_time", "direction", "exit_reason", "pnl_net",
    ]

    for i, (a, b) in enumerate(zip(trades_a, trades_b)):
        for field in compare_fields:
            va, vb = a.get(field, ""), b.get(field, "")
            if va != vb:
                result["match"] = False
                result["mismatches"].append(
                    f"Trade {i+1} field '{field}': '{va}' vs '{vb}'"
                )

    return result
