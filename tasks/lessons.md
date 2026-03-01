# Lessons and Patterns

## Architecture Decisions
- [x] Initial design focuses on modularity, separating engines (Vectorized vs. Event-Driven) to support different levels of fidelity and performance.

## Bias Mitigation
- [ ] Implement point-in-time data handling to prevent look-ahead bias.
- [ ] Use realistic slippage and commission models to avoid over-optimistic results.

## Testing Strategy
- [ ] Unit test each performance metric individually against known benchmarks.
- [ ] Use mock data to verify event sequence in the event-driven engine.
