# Advantage Actor-Critic Statistical Arbitrage for Intraday Electricity Markets

Replication of the reinforcement-learning core of **Demir, Kok & Paterakis (2023)** —
[*Statistical arbitrage trading across electricity markets using advantage actor-critic methods*](https://doi.org/10.1016/j.segan.2023.101023)
(Sustainable Energy, Grids and Networks 34, 101023).

The paper trades across three markets — the day-ahead market (DAM), continuous intraday market
(CID) and balancing market (BAL): a rule-based agent opens a position on the DAM, and a
synchronous advantage actor-critic (A2C) agent then manages that position on the CID tick by
tick, closing any leftover position on the BAL. One episode covers a single hourly delivery
contract's CID trading session; the agent chooses BUY / SELL / HOLD at every order-book
revision.

This project is a sibling of [`reinforce_threshold_policy`](../reinforce_threshold_policy)
(Bertrand & Papavasiliou 2020) and originally reused its synthetic CIM order-book + D-1 auction
dataset as-is; `scripts/data_generation/generate_synthetic_data.py` now generates all three
markets' data (CIM order book, D-1 auction curves, BAL take/feed prices) itself, using the same
CSV schema and train/test split convention as that sibling project (see
["Data"](#data) below for why it was rewritten). See
["Scope & simplifications"](#scope--simplifications) below for exactly what was and wasn't
replicated.

## Project structure

```
project_root/
  data/
    train/
      intraday_auction_curves.csv   # D-1 auction curves (2023-01-01, 181 days, seed=42) — generated, not tracked
      cim_order_book.csv            # CIM order book — generated, not tracked
      balancing_prices.csv          # quarter-hourly BAL take/feed prices — generated, not tracked
    test/
      intraday_auction_curves.csv   # D-1 auction curves (2023-08-01, 31 days, seed=123) — generated, not tracked
      cim_order_book.csv            # CIM order book — generated, not tracked
      balancing_prices.csv          # quarter-hourly BAL take/feed prices — generated, not tracked

  scripts/
    data_generation/
      generate_synthetic_data.py    # synthesise CIM + auction + BAL data for one split (--split train|test)
    data_plots/
      plot_auction_curve.py         # visualise D-1 auction MID curve + regimes
    check_data.py                   # validate a dataset

  src/
    data_loader.py       # load_all(), build_day_index(), day_auction_mids() — unchanged from sibling
    balancing.py           # load_split_balancing(), bal_bench(), rolling_neighbor_ptake() — BAL data access
    contracts.py             # list_contracts() — flattens build_day_index() into one episode per delivery hour
    dam_policy.py               # pdam / VWAP-BENCH proxies + rule-based DAM position opener
    rewards.py                    # buy/sell/hold reward functions (Eqs. 5-7)
    environment.py                  # ContractCIDEnv — one CID trading session per episode
    actor_critic.py                    # ActorCriticNet — two-headed shared network (Fig. 4, §4.6)
    a2c_trainer.py                        # A2CAgent, epsilon/gamma schedules, behaviour-cloning rules (Eqs. 8-11)
    parallel_worker.py                      # fork-based synchronous multi-worker train/eval episode functions
    benchmarks.py                             # HOLD and PRE-BA (Eq. 14) rule-based baselines
    training_logger.py                          # per-round metrics + training plots
    eval_plots.py                                 # six diagnostic evaluation figures

  outputs/
    runs/                # timestamped run directories (model.pt, hparams.txt,
                         #   training/ plots, eval/ plots) — not tracked
    data_plots/          # auction curve visualisations (.gitkeep tracked)

  train.py               # training entry point
  test.py                # quick greedy evaluation (rewards only)
  evaluate.py             # full evaluation with diagnostic plots + benchmarks
```

## Setup

```bash
python3 -m venv venv
venv/bin/pip install numpy pandas torch matplotlib scipy
```

## Data

All three markets' data (`cim_order_book.csv`, `intraday_auction_curves.csv`,
`balancing_prices.csv`) are synthesised by a single script, one data split at a time:

```bash
venv/bin/python3 scripts/data_generation/generate_synthetic_data.py --split train
venv/bin/python3 scripts/data_generation/generate_synthetic_data.py --split test
venv/bin/python3 scripts/check_data.py train
venv/bin/python3 scripts/check_data.py test
```

Same CSV schema, and the same train/test split convention, as `reinforce_threshold_policy`
(5 price levels/side CIM order book at 1-minute ticks, 10-level D-1 auction curves, train =
2023-01-01, 181 days, seed=42; test = 2023-08-01, 31 days, seed=123).

No `.csv` files are tracked in git (see `.gitignore`) — after cloning, run both commands above
before training or evaluating.

### Why a single custom generator instead of reusing the sibling project's data

This project originally copied `reinforce_threshold_policy`'s CIM/auction data verbatim and only
generated the (new, third-market) BAL prices itself. That approach drew the auction price, every
CID tick's price, and the BAL settlement price from **independent** noise around the same hourly
base price. A trained A2C agent collapsed to an always-HOLD policy, and root-causing it (rolling
out the paper's own Eq. (8)-(11) behaviour-cloning rules deterministically on real training
contracts) showed why: even those "expert" rules produced a non-HOLD action on **less than 1%**
of ticks, because the crossing conditions they check (`pa_t < pdam`, `pb_t > pfeed`, etc.) almost
never held when every market's price was just independent noise around one shared mean.

`generate_synthetic_data.py` instead gives every hourly contract **one continuous mean-reverting
process** (Ornstein-Uhlenbeck / AR(1) around the hourly base price, half-life 180 minutes,
stationary std 0.4 EUR/MWh) running from the D-1 15:00 auction gate-closure through physical
delivery: the auction locks in a sample of it at t=0, the CID mid-price is the same process
sampled every tick through the session, and the BAL reference is its value at the moment of
delivery (plus the same asymmetric take/feed premium/discount as before). A pure (non-reverting)
random walk was tried first and rejected — without mean reversion a session's price only ever
drifts in one direction, so within a single session only BUY or only SELL ever becomes
profitable, never both, and divergence from `pdam` grows unboundedly with session length instead
of settling into a realistic, bounded range. The mean-reverting version was calibrated
empirically (checked across several random seeds) so the same Eq. (8)-(11) crossing conditions
now fire on roughly 15-20% of ticks — genuinely tradeable opportunities, without being a
near-permanent one.

## Train

```bash
# Default: 200 days, 100 synchronisation rounds x 8 synchronous workers (paper's emax=100, W=8)
venv/bin/python3 train.py

# Smoke test
venv/bin/python3 train.py --days 2 --rounds 4 --workers 2

# Custom learning rate / output path / seed
venv/bin/python3 train.py --lr 5e-4 --out outputs/models/my_agent.pt --seed 7
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--days` | 200 | Training days |
| `--rounds` | 100 | Synchronisation rounds (paper's `emax`) |
| `--workers` | 8 | Synchronous workers per round (paper's `W`); total episodes = rounds x workers |
| `--lr` | 0.003 | Adam learning rate (paper's `beta`, §5.4.2) |
| `--n1` / `--n2` | 216 / 193 | Actor-critic hidden layer sizes (paper §5.4.2) |
| `--vmax` / `--vmin` | 10 / -10 | Position limits (MWh) |
| `--qhigh` | 50 | Max cumulative bought/sold quantity per contract (MWh) |
| `--pnl-low` / `--pnl-high` | -5000 / 10000 | PnL scaling range for the state features |
| `--eps-start` / `--eps-end` | 0.9 / 0.01 | Behaviour-cloning/exploration epsilon schedule |
| `--gamma-start` / `--gamma-end` | 0.29 / 0.9999 | Discount-factor annealing schedule |
| `--out` | `outputs/runs/<timestamp>/model.pt` | Checkpoint path |
| `--seed` | None | RNG seed |

Each run creates a timestamped directory under `outputs/runs/` containing `model.pt`,
`hparams.txt`, and training plots (`training/`).

## Evaluate

**Quick evaluation** (greedy policy, per-contract rewards + summary stats):
```bash
venv/bin/python3 test.py --model outputs/runs/<timestamp>/model.pt
venv/bin/python3 test.py --model outputs/runs/<timestamp>/model.pt --days 10
```

**Full evaluation** (greedy policy + 6 diagnostic plots + benchmarks):
```bash
venv/bin/python3 evaluate.py --model outputs/runs/<timestamp>/model.pt

# Include the PRE-BA rule-based benchmark (Eq. 14); HOLD is always computed
venv/bin/python3 evaluate.py --model outputs/runs/<timestamp>/model.pt --with-pre-ba

# Parallel evaluation across CPU cores
venv/bin/python3 evaluate.py --model outputs/runs/<timestamp>/model.pt --workers 8
```

Prints a paper-Table-7-style summary (traded quantity, PnL, %PnL>0, profit-to-deviation PD,
profit-to-trade PT) for A2C vs. HOLD vs. PRE-BA, and saves six figures:

1. Performance distribution — A2C vs HOLD vs PRE-BA PnL histograms
2. Cumulative PnL across test contracts (paper Fig. 9)
3. Traded quantity distribution by delivery hour (paper Fig. 10)
4. PnL distribution by delivery hour (paper Fig. 11)
5. Example contract tick traces — bid/ask + BUY/SELL markers
6. Action distribution across the test set

## Scope & simplifications

This replicates the paper's **A2C trading agent (§4)** only — state/action/reward spec, the
two-headed actor-critic network, and synchronous multi-worker training with behaviour cloning —
the same scope precedent set by `reinforce_threshold_policy` for its own paper. It does **not**
implement:

- **DAM/vwap price forecasting (§3).** The paper trains 2NN/2CNN_NN networks with
  autoencoder/VAE/GAN data augmentation for DAM prices, and a LASSO/RF/GB/DNN ensemble for CID
  vwap, validated against real ENTSO-E / Scholt Energy data. We don't have that data, so:
  - `pdam` is proxied by the D-1 auction MID price for the delivery hour
    (`src/data_loader.day_auction_mids`).
  - The vwap forecast used only to pick the DAM long/short direction is proxied by the paper's
    *own* VWAP-BENCH baseline (Eq. 13): a causal 2-day trailing average of the realized CID
    price for that delivery hour. See `src/dam_policy.py` for the full reasoning, including why
    "realized CID price" is itself approximated by the order-book mid-price time-average (our
    synthetic data records resting quotes, not an executed trade tape).
- **Real (i.e. historical) balancing-market prices.** `data/{train,test}/balancing_prices.csv` is
  a *synthesized* third market — quarter-hourly take/feed prices sharing the same diurnal price
  pattern as the auction/CIM generators but drawn independently, with an asymmetric imbalance
  premium/discount (`take_price > reference > feed_price`, see
  `scripts/data_generation/generate_*_balancing.py`). It is not derived from a contract's own CID
  session, so BAL settlement (`src/environment.py`'s terminal branch) is genuinely a third,
  independent price series — it's just not backed by real ENTSO-E/Scholt Energy balancing data
  the way the paper's is. Mid-episode BAL-referencing state features and the Eq. (10)
  behaviour-cloning rule only ever see *causal* trailing benchmarks of past contracts'
  settlement prices (`src/balancing.py`'s `bal_bench`/`rolling_neighbor_ptake`), never the
  current contract's own (future) settlement — that would be lookahead.
- **Monthly rolling retraining (§5.4.2).** The paper retrains every month on a rolling window;
  this project uses a single fixed 200-day train / 31-day test split, trained once, consistent
  with `reinforce_threshold_policy`'s simpler setup.
- **The A3C benchmark.** Only A2C is implemented; HOLD and PRE-BA (Eq. 14) are included since
  they're cheap rule-based comparators evaluated the same way the paper does.
- **The `tmax` intra-episode update horizon (Algorithm 1).** Our per-contract episodes are never
  longer than ~1890 ticks (the last delivery hour's ~31.5h CID session at 1-minute resolution),
  always shorter than the paper's `tmax=2906`, so the "update the global network every `tmax`
  steps, possibly mid-episode" mechanic never actually triggers -- every worker always plays a
  full episode to its true terminal state before its gradient is computed. This is mathematically
  identical to the paper's algorithm in our setting, just simpler to state.
- **The `trandom` behaviour-cloning window (Algorithm 1, lines 15-19).** The paper picks a random
  start point within the episode: before it, the worker explores by sampling from its own policy;
  after it, the worker clones its assigned rule. Since `tmax` exceeds our episode lengths, that
  window would cover nearly the whole episode regardless of where it starts, so we replace it with
  an unconditional 50/50 split between cloning and policy-sampled exploration whenever the epsilon
  roll triggers (`src/a2c_trainer.py`'s `select_action`) -- both of the paper's non-greedy
  behaviours stay reachable, just not tied to a specific time window.

All of these are called out inline in the relevant source files (`src/dam_policy.py`,
`src/environment.py`, `src/a2c_trainer.py`) as well.

### Fixed during a paper-fidelity audit

Repeated systematic line-by-line checks against the paper (Eqs. 1-14, Table 1, Algorithm 1,
§5.4.2) found and fixed six real gaps beyond the deliberate scope simplifications above:

- **Trading cost.** Eq. (1)'s `TC = 0.116 EUR/MWh x (total bought + total sold)` term was missing
  entirely from the PnL calculation (`src/environment.py`). It doesn't affect training -- the
  reward functions (Eqs. 5-7) never reference cost or quantity -- but it was inflating every
  reported PnL, disproportionately for the more active strategies (A2C, PRE-BA) versus HOLD.
- **Learning rate.** `train.py --lr` defaulted to `1e-3`; the paper's §5.4.2 hyperparameter is
  `beta = 0.003`.
- **The "Quantity" metric in Table 7 -- DAM leg.** Worked out from the paper's own numbers: HOLD's
  reported quantity is exactly `vmax` per contract despite HOLD making zero CID trades, meaning
  "Quantity" = `|v0|` (the DAM leg, always a full `vmax`/`vmin` position) + CID quantity, not CID
  quantity alone. `evaluate.py` was hardcoding HOLD's quantity to `0` and incorrectly reusing
  A2C's traded quantity when reporting PRE-BA's PT/PD stats; `src/benchmarks.py`'s
  `run_hold`/`run_pre_ba` now track and return their own quantity, DAM leg included.
- **The Eq. (11) threshold clone rule used the wrong session extremes.** `make_threshold_clone_rule`
  (workers 4-8) computed its would-be buy/sell reward via `ctx.get("pa_high", ctx["pa_low"])` and
  `ctx.get("pb_low", ctx["pb_high"])` -- fallbacks that fired on every call, since `rule_context()`
  never actually included `pa_high`/`pb_low`, silently substituting `pa_low` for `pa_high` and
  `pb_high` for `pb_low`. This only corrupted the cloning *decision* for those 5 workers, not the
  real training reward (computed independently and correctly in `ContractCIDEnv.step()`).
  `rule_context()` now returns the real `pa_high`/`pb_low`.
- **The "Quantity" metric in Table 7 -- gross vs. arbitraged CID volume.** The DAM-leg fix above
  measured the CID leg as the *gross* sum of buy + sell volume (`sum(qa_t) + sum(qb_t)`). The
  paper, however, explicitly names this quantity in its own Section 2.2 / Nomenclature: `Sigma q`,
  "total arbitraged quantity", is defined as `min(sum(qa_t), sum(qb_t))` -- the netted,
  round-tripped volume, which is smaller than the gross sum whenever a contract both buys and
  sells. `src/benchmarks.py` and `src/parallel_worker.py` now use `min(cum_bought, cum_sold)` for
  the CID leg of "Quantity" (and hence PT = PnL / Quantity) for all three agents (A2C, HOLD,
  PRE-BA), matching the paper's own formula instead of an inferred gross convention.
- **PRE-BA (Eq. 14) averaged in the current tick's own price.** `run_pre_ba`'s trailing 30-tick
  window included index `t` itself (comparing `pa_t` against an average that contained `pa_t`),
  contradicting the paper's own framing of PRE-BA as trading against "the previous best bid and
  best ask prices" (§5.4.4). The window now looks strictly backwards, excluding the current tick
  (falling back to a same-tick no-op comparison, i.e. HOLD, at the very first tick of a session
  where no history yet exists).

## Reference

Demir, S., Kok, K., & Paterakis, N. G. (2023). Statistical arbitrage trading across electricity
markets using advantage actor-critic methods. *Sustainable Energy, Grids and Networks*, 34,
101023. https://doi.org/10.1016/j.segan.2023.101023
