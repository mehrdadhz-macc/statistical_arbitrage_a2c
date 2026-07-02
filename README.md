# Advantage Actor-Critic Statistical Arbitrage for Intraday Electricity Markets

Replication of the reinforcement-learning core of **Demir, Kok & Paterakis (2023)** —
*Statistical arbitrage trading across electricity markets using advantage actor-critic methods*
(Sustainable Energy, Grids and Networks 34, 101023).

The paper trades across three markets — the day-ahead market (DAM), continuous intraday market
(CID) and balancing market (BAL): a rule-based agent opens a position on the DAM, and a
synchronous advantage actor-critic (A2C) agent then manages that position on the CID tick by
tick, closing any leftover position on the BAL. One episode covers a single hourly delivery
contract's CID trading session; the agent chooses BUY / SELL / HOLD at every order-book
revision.

This project is a sibling of [`reinforce_threshold_policy`](../reinforce_threshold_policy)
(Bertrand & Papavasiliou 2020) and **reuses its synthetic CIM order-book + D-1 auction dataset
as-is** (same schema, same data-loading code, same train/test split) and adds a **third,
independently-generated synthetic dataset for the balancing market** so the DAM-CID-BAL
structure is real rather than degenerating to two markets. See
["Scope & simplifications"](#scope--simplifications) below for exactly what was and wasn't
replicated.

## Project structure

```
project_root/
  data/
    train/
      intraday_auction_curves.csv   # D-1 auction curves (Jan-Jun 2023, seed=42) — copied, not tracked
      cim_order_book.csv            # CIM order book — copied, not tracked
      balancing_prices.csv          # quarter-hourly BAL take/feed prices — generated here, tracked
    test/
      intraday_auction_curves.csv   # D-1 auction curves (Aug 2023, seed=123) — copied, not tracked
      cim_order_book.csv            # CIM order book — copied, not tracked
      balancing_prices.csv          # quarter-hourly BAL take/feed prices — generated here, tracked

  scripts/
    data_generation/
      generate_train_data.py        # synthesise training CIM + auction data (unchanged from sibling)
      generate_test_data.py         # synthesise test CIM + auction data (unchanged from sibling)
      generate_train_balancing.py   # synthesise training BAL take/feed prices (new third market)
      generate_test_balancing.py    # synthesise test BAL take/feed prices (new third market)
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

Two of the three markets' data (`cim_order_book.csv`, `intraday_auction_curves.csv`) are copied
verbatim from `reinforce_threshold_policy/data/` — same CIM order book (5 price levels/side,
1-minute ticks) and D-1 auction curves (10 levels/side), same 200-day train / 31-day test split.
See that project's `scripts/data_generation/` for how they were generated, or regenerate them
locally:

```bash
venv/bin/python3 scripts/data_generation/generate_train_data.py
venv/bin/python3 scripts/data_generation/generate_test_data.py
venv/bin/python3 scripts/check_data.py train
venv/bin/python3 scripts/check_data.py test
```

The third market (`balancing_prices.csv`) is new to this project — quarter-hourly BAL take/feed
prices, generated so their `delivery_start` values line up exactly with the existing CIM/auction
data (same day set, same diurnal base price pattern, independent noise draws):

```bash
venv/bin/python3 scripts/data_generation/generate_train_balancing.py
venv/bin/python3 scripts/data_generation/generate_test_balancing.py
```

Unlike the multi-GB CIM order books, `balancing_prices.csv` is small (~1MB) and **is tracked in
git** — no regeneration needed after cloning.

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
| `--lr` | 1e-3 | Adam learning rate |
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

All of these are called out inline in the relevant source files (`src/dam_policy.py`,
`src/environment.py`) as well.

## Reference

Demir, S., Kok, K., & Paterakis, N. G. (2023). Statistical arbitrage trading across electricity
markets using advantage actor-critic methods. *Sustainable Energy, Grids and Networks*, 34,
101023. https://doi.org/10.1016/j.segan.2023.101023
