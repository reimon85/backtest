# EEVS — EaglesEye Validation Standard

Deterministic, auditable WFA validation system. Eliminates discrepancies between AI tools (Claude/Gemini) by enforcing a single source of truth for indicator computation, simulation rules, and acceptance criteria.

## Quick start

```bash
# Run WFA validation for ZNas100
python -m eevs run --strategy znas100

# Compare two audit CSVs
python -m eevs verify audit_a.csv audit_b.csv

# Override data file
python -m eevs run --strategy znas100 --data data/my_custom_file.csv
```

## Architecture

| Module | Purpose |
|--------|---------|
| `config.py` | Immutable `StrategyConfig`, `CostModel`, `COST_TABLE` |
| `data_factory.py` | Pre-compute indicators (anti-look-ahead). Strategy only reads. |
| `simulator.py` | Bar-by-bar engine: `next_open` entry, worst-case TP/SL collision |
| `audit.py` | Write/verify audit CSVs with SHA-256 hashing |
| `report.py` | WFA report: monthly table, compound metrics, acceptance, Monte Carlo |
| `runner.py` | CLI entry point |
| `signals/` | Signal functions (one per strategy) |

## Deterministic rules (hardcoded)

- **Entry**: `next_open` (bar after signal)
- **Fill price**: `next_open ± cost_per_side`
- **TP/SL collision**: worst-case (SL wins)
- **Costs**: per PRODUCTION_STANDARD §4
- **Indicators**: causal (rolling/ewm), daily with `shift(1)`
- **Z-Score ddof**: 1 (explicit)

## Adding a new strategy

1. Create `eevs/signals/my_strategy.py` with:
   - `MY_PARAMS` dict with production parameters
   - `make_config(data_path) -> StrategyConfig`
   - `signal_fn(row, state) -> "LONG" | "SHORT" | None`
2. Register in `runner.py` `_register_strategies()`
3. Run: `python -m eevs run --strategy my_strategy`

## Output

Audit CSVs are written to `eevs_output/` (gitignored). Each CSV contains:
- Metadata header (strategy, config hash, data hash, timestamp)
- One row per trade with full audit trail
- SHA-256 of the file for integrity verification
