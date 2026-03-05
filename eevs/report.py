"""EEVS report — WFA report generator with acceptance criteria.

Reuses logic from src/robustness.py acceptance() and
research/validate_znas100_wfa.py compound/fold_table/monte_carlo.
"""

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .simulator import TradeRecord


# ── Monthly fold table ──────────────────────────────────────────────────────


def fold_table(trades: list[TradeRecord]) -> pd.DataFrame:
    """Per-month breakdown: trades, TP, SL, WR%, PnL_pts, PF."""
    if not trades:
        return pd.DataFrame()

    rows = []
    monthly: dict[str, list[TradeRecord]] = {}
    for t in trades:
        monthly.setdefault(t.month, []).append(t)

    for month in sorted(monthly):
        group = monthly[month]
        pnls = [t.pnl_net for t in group]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        gp = sum(wins)
        gl = abs(sum(losses))
        pf = gp / gl if gl > 0 else float("inf")
        rows.append({
            "mes": month,
            "trades": len(pnls),
            "TP": len(wins),
            "SL": len(losses),
            "WR%": round(len(wins) / len(pnls) * 100, 1) if pnls else 0.0,
            "PnL_pts": round(sum(pnls), 1),
            "PF": round(pf, 3),
        })

    return pd.DataFrame(rows).set_index("mes")


# ── Compound metrics ────────────────────────────────────────────────────────


def compound_metrics(trades: list[TradeRecord]) -> dict:
    """Aggregate metrics across all trades (PRODUCTION_STANDARD §5)."""
    if not trades:
        return {}

    pnls = [t.pnl_net for t in trades]
    pnls_eq = [t.pnl_eq for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gp = sum(wins)
    gl = abs(sum(losses))
    pf = gp / gl if gl > 0 else float("inf")

    # Monthly aggregation
    monthly_pts: dict[str, float] = {}
    monthly_eq: dict[str, float] = {}
    for t in trades:
        monthly_pts[t.month] = monthly_pts.get(t.month, 0.0) + t.pnl_net
        monthly_eq[t.month] = monthly_eq.get(t.month, 0.0) + t.pnl_eq

    months_pos = sum(1 for v in monthly_pts.values() if v > 0)
    total_months = len(monthly_pts)

    # Quarterly PF
    quarterly_pf: dict[str, float] = {}
    q_buckets: dict[str, dict[str, float]] = {}
    for t in trades:
        dt = pd.Timestamp(t.month + "-01")
        q = f"{dt.year}Q{(dt.month - 1) // 3 + 1}"
        if q not in q_buckets:
            q_buckets[q] = {"profit": 0.0, "loss": 0.0}
        if t.pnl_net > 0:
            q_buckets[q]["profit"] += t.pnl_net
        else:
            q_buckets[q]["loss"] += abs(t.pnl_net)
    for q, v in q_buckets.items():
        quarterly_pf[q] = v["profit"] / v["loss"] if v["loss"] > 0 else float("inf")

    # Max drawdown (equity curve)
    cum_eq = np.cumsum(pnls_eq)
    peak_eq = np.maximum.accumulate(cum_eq)
    max_dd = float((cum_eq - peak_eq).min()) if len(cum_eq) > 0 else 0.0

    return {
        "pf_oos": pf,
        "pnl_oos_pts": sum(pnls),
        "pnl_oos_eq": sum(pnls_eq),
        "total_trades": len(pnls),
        "wr_oos": len(wins) / len(pnls) if pnls else 0.0,
        "worst_month_pts": min(monthly_pts.values()) if monthly_pts else 0.0,
        "worst_month_eq": min(monthly_eq.values()) if monthly_eq else 0.0,
        "best_month_pts": max(monthly_pts.values()) if monthly_pts else 0.0,
        "best_month_eq": max(monthly_eq.values()) if monthly_eq else 0.0,
        "months_pos": months_pos,
        "total_months": total_months,
        "months_pos_pct": months_pos / total_months if total_months > 0 else 0.0,
        "quarterly_pf": quarterly_pf,
        "max_dd": max_dd,
    }


# ── Acceptance criteria (PRODUCTION_STANDARD §5) ───────────────────────────


def acceptance(m: dict) -> dict:
    """Evaluate PRODUCTION_STANDARD acceptance criteria."""
    oblig = {
        "PF OOS > 1.05":           m["pf_oos"] > 1.05,
        "PnL OOS > 0":             m["pnl_oos_pts"] > 0,
        "Trades OOS >= 50":        m["total_trades"] >= 50,
        "Peor mes equity > -5%":   m["worst_month_eq"] > -0.05,
    }
    consist = {
        "Meses+ >= 50%":           m["months_pos_pct"] >= 0.50,
        "WR OOS > 35%":            m["wr_oos"] > 0.35,
        "Ningún trim. PF < 0.5":   all(v >= 0.5 for v in m["quarterly_pf"].values()),
    }
    all_ok = all(oblig.values())
    n_cons = sum(consist.values())

    if all_ok and n_cons >= 2:
        verdict = "PRODUCCION_ESTANDAR  (riesgo 1%)"
    elif all_ok and n_cons == 1:
        verdict = "PRODUCCION_CONTROLADA (riesgo 0.5% + revision mensual)"
    elif all_ok:
        verdict = "NO_LISTA — revisar logica o parametros"
    else:
        verdict = "BLOQUEADA — criterios obligatorios no cumplidos"

    return {
        "obligatory": oblig,
        "consistency": consist,
        "n_consistency": n_cons,
        "verdict": verdict,
        "all_oblig_ok": all_ok,
    }


# ── Monte Carlo ─────────────────────────────────────────────────────────────


def monte_carlo(
    pnls_eq: list[float],
    n_sims: int = 5000,
    ruin_threshold: float = -0.30,
) -> dict:
    """Shuffled Monte Carlo simulation for robustness analysis."""
    rng = np.random.default_rng(42)
    arr = np.array(pnls_eq)
    n = len(arr)

    def equity_curve(returns: np.ndarray) -> np.ndarray:
        return np.cumprod(1.0 + returns)

    def max_drawdown(eq: np.ndarray) -> float:
        peak = np.maximum.accumulate(eq)
        return float((eq / peak - 1.0).min())

    eq_real = equity_curve(arr)
    actual_dd = max_drawdown(eq_real)
    actual_eq = float(eq_real[-1] - 1.0)

    # Outlier robustness: remove top 5% trades
    top_k = max(1, int(n * 0.05))
    arr_no_top = np.sort(arr)[:-top_k]
    eq_no_top = equity_curve(arr_no_top)
    pnl_no_top = float(eq_no_top[-1] - 1.0)

    all_max_dd = np.empty(n_sims)
    all_final = np.empty(n_sims)
    ruin_count = 0

    for i in range(n_sims):
        shuffled = arr[rng.permutation(n)]
        eq = equity_curve(shuffled)
        dd = max_drawdown(eq)
        all_max_dd[i] = dd
        all_final[i] = eq[-1] - 1.0
        if dd <= ruin_threshold:
            ruin_count += 1

    return {
        "n_sims": n_sims,
        "risk_of_ruin": ruin_count / n_sims * 100,
        "dd_median": float(np.median(all_max_dd)),
        "dd_p95": float(np.percentile(all_max_dd, 5)),
        "dd_p99": float(np.percentile(all_max_dd, 1)),
        "final_median": float(np.median(all_final)),
        "final_p5": float(np.percentile(all_final, 5)),
        "final_p95": float(np.percentile(all_final, 95)),
        "actual_max_dd": actual_dd,
        "actual_final": actual_eq,
        "pnl_no_top5": pnl_no_top,
        "n_removed": top_k,
    }


# ── Report generator ───────────────────────────────────────────────────────


def generate_report(trades: list[TradeRecord], config: StrategyConfig) -> dict:
    """Generate complete WFA report with metrics, criteria, and verdict."""
    tbl = fold_table(trades)
    m = compound_metrics(trades)
    acc = acceptance(m)
    pnls_eq = [t.pnl_eq for t in trades]
    mc = monte_carlo(pnls_eq) if len(pnls_eq) >= 10 else {}

    return {
        "config": config,
        "fold_table": tbl,
        "compound": m,
        "acceptance": acc,
        "monte_carlo": mc,
        "total_trades": len(trades),
    }


def print_report(report: dict) -> None:
    """Print formatted WFA report to stdout."""
    SEP = "=" * 72
    LINE = "-" * 72
    config = report["config"]
    m = report["compound"]
    acc = report["acceptance"]
    mc = report.get("monte_carlo", {})
    cost = config.cost_model

    print(SEP)
    print(f"  WFA VALIDATION — {config.name} v{config.version}   (EEVS v1.0)")
    print(SEP)

    print(f"\nActivo:    {config.asset} ({config.timeframe})")
    print(f"Dataset:   {config.data_path}")
    print(f"Warmup:    {config.warmup_bars} barras")
    print(f"Entry:     {config.entry_mode}")
    print(f"Costes:    spread={cost.spread} + slippage={cost.slippage} + com={cost.commission}"
          f" = {cost.cost_per_side}/lado ({cost.round_trip} RT)")
    print(f"Gestion:   TP={config.tp_points}pts / SL={config.sl_points}pts")

    print(f"\n{LINE}")
    print("RESULTADOS POR MES (OOS compuesto)")
    print(report["fold_table"].to_string())

    print(f"\n{LINE}")
    print("METRICAS COMPUESTAS OOS")
    print(f"  PF OOS:                   {m['pf_oos']:.3f}")
    print(f"  PnL OOS (pts):            {m['pnl_oos_pts']:+.1f} pts")
    print(f"  PnL OOS (equity 1%R):     {m['pnl_oos_eq']*100:+.2f}%")
    print(f"  Trades totales:           {m['total_trades']}")
    print(f"  WR OOS:                   {m['wr_oos']*100:.1f}%")
    print(f"  Mejor mes (pts):         +{m['best_month_pts']:.1f} pts")
    print(f"  Peor mes  (pts):          {m['worst_month_pts']:.1f} pts")
    print(f"  Peor mes  (equity 1%R):   {m['worst_month_eq']*100:.2f}%")
    print(f"  MaxDD curva (equity 1%R): {m['max_dd']*100:.2f}%")
    print(f"  Meses positivos:          {m['months_pos']}/{m['total_months']}"
          f"  ({m['months_pos_pct']*100:.0f}%)")

    print("\n  PF trimestral:")
    for q, pf_q in sorted(m["quarterly_pf"].items()):
        flag = "!  " if pf_q < 0.5 else "ok "
        print(f"    {flag}{q}: {pf_q:.3f}")

    print(f"\n{LINE}")
    print("CRITERIOS OBLIGATORIOS (deben cumplirse TODOS)")
    for crit, ok in acc["obligatory"].items():
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {crit}")

    print("\nCRITERIOS DE CONSISTENCIA (se requieren >= 2 de 3)")
    for crit, ok in acc["consistency"].items():
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {crit}")
    print(f"       Pasados: {acc['n_consistency']}/3")

    print(f"\n{SEP}")
    print(f"  VEREDICTO: {acc['verdict']}")
    print(SEP)

    if mc:
        print(f"\n{LINE}")
        print(f"MONTE CARLO  ({mc['n_sims']} simulaciones — shuffling de trades OOS)")
        print(LINE)
        print(f"  Equity final real:  {mc['actual_final']*100:+.2f}%")
        print(f"  MaxDD real:         {mc['actual_max_dd']*100:.2f}%")
        print(f"  MC mediana:         {mc['final_median']*100:+.2f}%")
        print(f"  MC p5/p95:          {mc['final_p5']*100:+.2f}% / {mc['final_p95']*100:+.2f}%")
        print(f"  DD mediana:         {mc['dd_median']*100:.2f}%")
        print(f"  DD p95/p99:         {mc['dd_p95']*100:.2f}% / {mc['dd_p99']*100:.2f}%")
        print(f"  Riesgo ruina:       {mc['risk_of_ruin']:.2f}%")
        print(f"  Sin top 5%:         {mc['pnl_no_top5']*100:+.2f}%"
              f"  ({'rentable' if mc['pnl_no_top5'] > 0 else 'NO rentable'})")
