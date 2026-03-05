"""EEVS runner — CLI entry point.

Usage:
    python -m eevs.runner run --strategy znas100
    python -m eevs.runner verify file_a.csv file_b.csv
"""

import argparse
import sys

from .config import StrategyConfig
from .data_factory import build_dataframe
from .simulator import simulate
from .audit import write_audit_csv, verify_audit
from .report import generate_report, print_report


# ── Strategy registry ───────────────────────────────────────────────────────

STRATEGIES: dict[str, tuple] = {}


def _register_strategies():
    """Lazy-load strategy modules to avoid circular imports."""
    from .signals.znas100 import make_config, signal_fn
    STRATEGIES["znas100"] = (make_config, signal_fn)


# ── Commands ────────────────────────────────────────────────────────────────


def cmd_run(args):
    """Execute WFA validation and generate audit CSV."""
    _register_strategies()

    if args.strategy not in STRATEGIES:
        print(f"Unknown strategy: {args.strategy}")
        print(f"Available: {', '.join(STRATEGIES.keys())}")
        sys.exit(1)

    make_config, signal_fn = STRATEGIES[args.strategy]
    config = make_config(data_path=args.data) if args.data else make_config()

    print(f"Loading data: {config.data_path}")
    df = build_dataframe(config)
    print(f"Dataset: {df.index[0]} -> {df.index[-1]} ({len(df)} bars)")
    print(f"Warmup: {config.warmup_bars} bars, Sim: {len(df) - config.warmup_bars} bars")

    print("Simulating...")
    trades = simulate(df, config, signal_fn)
    print(f"Trades closed: {len(trades)}")

    if not trades:
        print("No trades generated. Check warmup, params, or data quality.")
        sys.exit(1)

    # Generate and print report
    report = generate_report(trades, config)
    print_report(report)

    # Write audit CSV
    output_dir = args.output or "eevs_output"
    filepath, sha = write_audit_csv(trades, config, output_dir)
    print(f"\nAudit CSV: {filepath}")
    print(f"SHA-256:   {sha}")


def cmd_verify(args):
    """Compare two audit CSVs trade by trade."""
    result = verify_audit(args.file_a, args.file_b)

    print(f"File A: {result['file_a']}  ({result['trades_a']} trades)")
    print(f"File B: {result['file_b']}  ({result['trades_b']} trades)")

    if result["match"]:
        print("\nRESULT: MATCH — all trades identical")
    else:
        print(f"\nRESULT: MISMATCH — {len(result['mismatches'])} differences found")
        for m in result["mismatches"][:20]:
            print(f"  - {m}")
        if len(result["mismatches"]) > 20:
            print(f"  ... and {len(result['mismatches']) - 20} more")


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        prog="eevs.runner",
        description="EEVS — EaglesEye Validation Standard",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Run WFA validation")
    p_run.add_argument("--strategy", "-s", required=True, help="Strategy name (e.g. znas100)")
    p_run.add_argument("--data", "-d", help="Override data file path")
    p_run.add_argument("--output", "-o", help="Output directory (default: eevs_output)")

    # verify
    p_verify = sub.add_parser("verify", help="Compare two audit CSVs")
    p_verify.add_argument("file_a", help="First audit CSV")
    p_verify.add_argument("file_b", help="Second audit CSV")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "verify":
        cmd_verify(args)


if __name__ == "__main__":
    main()
