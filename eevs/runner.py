"""EEVS runner — CLI entry point.

Usage:
    python -m eevs run --strategy znas100
    python -m eevs optimize --strategy znas100 --tp 300,350,400 --sl 200,250,300
    python -m eevs verify file_a.csv file_b.csv
"""

import argparse
import sys
import copy
from rich.console import Console
from rich.table import Table

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


def cmd_optimize(args):
    """Run grid search optimization over production simulator."""
    _register_strategies()
    console = Console()

    if args.strategy not in STRATEGIES:
        print(f"Unknown strategy: {args.strategy}")
        sys.exit(1)

    make_config, signal_fn = STRATEGIES[args.strategy]
    base_config = make_config(data_path=args.data) if args.data else make_config()

    # Parse ranges
    tp_range = [float(x) for x in args.tp.split(",")] if args.tp else [base_config.params["tp_points"]]
    sl_range = [float(x) for x in args.sl.split(",")] if args.sl else [base_config.params["sl_points"]]

    print(f"Loading data: {base_config.data_path}")
    df = build_dataframe(base_config)
    
    results = []
    
    table = Table(title=f"EEVS Optimization — {args.strategy}")
    table.add_column("TP", justify="right")
    table.add_column("SL", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("WR%", justify="right")
    table.add_column("PnL Pts", justify="right", style="green")
    table.add_column("PF", justify="right", style="bold cyan")

    with console.status("[bold green]Running Grid Search...") as status:
        for tp in tp_range:
            for sl in sl_range:
                # Create a temporary config variant
                cfg = copy.deepcopy(base_config)
                cfg.params["tp_points"] = tp
                cfg.params["sl_points"] = sl
                
                trades = simulate(df, cfg, signal_fn)
                if not trades:
                    continue
                
                report = generate_report(trades, cfg)
                m = report["compound"]
                
                results.append({
                    "tp": tp, "sl": sl, "trades": m["total_trades"],
                    "wr": m["wr_oos"], "pnl": m["pnl_pts"], "pf": m["pf_oos"]
                })

    # Sort by PF
    sorted_res = sorted(results, key=lambda x: x["pf"], reverse=True)
    
    for r in sorted_res:
        table.add_row(
            f"{r['tp']:.1f}", f"{r['sl']:.1f}", str(r['trades']),
            f"{r['wr']:.1f}%", f"{r['pnl']:+.1f}", f"{r['pf']:.3f}"
        )

    console.print(table)


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
        prog="eevs",
        description="EEVS — EaglesEye Validation Standard",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Run WFA validation")
    p_run.add_argument("--strategy", "-s", required=True, help="Strategy name")
    p_run.add_argument("--data", "-d", help="Override data file path")
    p_run.add_argument("--output", "-o", help="Output directory")

    # optimize
    p_opt = sub.add_parser("optimize", help="Run grid search optimization")
    p_opt.add_argument("--strategy", "-s", required=True, help="Strategy name")
    p_opt.add_argument("--tp", help="TP range (comma separated)")
    p_opt.add_argument("--sl", help="SL range (comma separated)")
    p_opt.add_argument("--data", "-d", help="Override data file path")

    # verify
    p_verify = sub.add_parser("verify", help="Compare two audit CSVs")
    p_verify.add_argument("file_a", help="First audit CSV")
    p_verify.add_argument("file_b", help="Second audit CSV")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "verify":
        cmd_verify(args)


if __name__ == "__main__":
    main()
