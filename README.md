# lppl_new

Fits the Log-Periodic Power Law Singularity (LPPLS) bubble model to daily
stock and crypto prices, turns the fits into bubble-confidence indicators,
backtests contrarian strategies on those indicators, and in production mode
writes daily buy/sell candidates scored by the backtest results.

## Model

For a window ending at `t2`, log price is fitted as

```
ln p(t) = a + |tc - t|^m * (b + c1 cos(w ln|tc - t|) + c2 sin(w ln|tc - t|))
```

The nonlinear parameters `(tc, m, w)` are found with Nelder-Mead from 15
random restarts; the linear parameters `(a, b, c1, c2)` are solved exactly
by least squares inside every cost evaluation (Filimonov & Sornette 2013).
Time is calendar days; price is min-max scaled `ln(Close)` per ticker.

For every date, 18 nested windows (120 bars shrinking to 30 in steps of 5)
are fitted. A fit "qualifies" when it passes the Sornette filter
(`m` in [0.01, 0.99], `w` in [2, 15], oscillations >= 2.5, damping >= 0.5,
`tc` inside a window-relative range) and has R^2 >= `FIT_R2_MIN`.

- `pos_conf` = qualified positive-bubble fits / positive-bubble fits
- `neg_conf` = the same for negative bubbles (crashes)
- `pos_tc` / `neg_tc` = median fitted critical time of the qualified fits
- `eps_norm` = median residual z-score of the qualified fits

## Strategies

Contrarian: `lppl_{p1}_{p2}` goes short when `pos_conf > p1` and long when
`neg_conf > p2`. `res_{tau}_{dmin}` goes long when `eps_norm <= -tau` for
`dmin` days and short when `>= +tau`. Signals act at the next open.

Exits, in priority order: opposite signal, 20% stop, `tc` plus 30 calendar
days, confidence gone for 5 bars (only when the trade has no `tc` target),
126-bar maximum hold, end of data (mark-to-market). The summary reports a
counter for each exit reason.

Testing mode runs the whole `(p1, p2)` grid and the residual grid.
Production mode runs the one `(p1, p2)` chosen per universe group in
`run_backtests` in `src/lib.rs`.

## Pipeline

```
ClickHouse (tiingo.usd / crypto)
  -> data/{testing|production}/{universe}.csv      prices, 5 years in testing
  -> fit/{testing|production}/fit_{universe}_{ticker}.csv   nested fits
  -> output[_crypto]/{testing|production}/{ticker}.parquet  signals + per-ticker stats
  -> performance/{stocks|crypto}_testing.csv       pooled metrics per strategy and universe
     performance/{LC,MC,SC,Micro}.csv              the same, split by group
  production only:
  -> performance/{stocks|crypto}_{buys,sells}_{date}.csv
  -> score/{stocks|crypto}_{date}.csv and the ClickHouse table lppl_score
```

Testing fit CSVs are deleted after the backtests (they run to ~10 GB per
group). The summary metrics are pooled over trades (not averaged over
tickers) and include `expectancy` (mean % per trade), `std_err`, `t_stat`,
`profit_factor` and `hit_ratio`; a strategy needs 20 trades to appear.

### Data screening

`read_price_file` drops data breaks before anything sees the prices: a
one-day close ratio outside [0.4, 2.5] or a close under $0.10 ends a
segment, and only the longest clean segment per ticker is kept (the latest
one in production). This removes unadjusted reverse splits, reused ticker
symbols and sub-penny rounding, which otherwise produce fake crashes for the
fitter and fake 1000%+ trades for the backtester. Crypto files are exempt.

## Running

```
cargo run --release -- <universe> <testing|production> [path] [resume]
```

- `universe`: a group (`LC`, `MC`, `SC`, `Micro`, `Stocks`, `Crypto`), a
  single universe (`LC1`, `SC3`, ...), or a comma-separated list.
- `path`: project root, default `$CLICKHOUSE_USER_PATH/rust_home/lppl_new`
  (`CLICKHOUSE_USER_PATH` defaults to `/Users/rogerbos`).
- `resume`: keep existing price, fit and output files and compute only what
  is missing. Without it the run deletes the files of the universes being
  run (other groups' files are kept, so groups can be run one at a time).

Environment: `PG` holds the ClickHouse password (user `roger`, database
`tiingo`, hosts in `src/clickhouse_mod.rs`).

Examples:

```
nohup cargo run --release -- Stocks testing > nohup.out 2>&1 &

# cargo run --release -- Stocks testing          # full five-year backtest, all 12 universes
# cargo run --release -- LC testing              # one group
# cargo run --release -- LC,MC testing x resume  # continue an interrupted run
# cargo run --release -- Stocks production       # today's buys/sells and scores
# cargo run --release -- Crypto production -->
**```

A full stock testing run takes several days on 10 cores; run it under
`nohup` or `tmux`. The fit and parquet schemas changed in October 2026, so
files from earlier runs cannot be resumed.

## Tuning constants

| Constant | File | Meaning |
|---|---|---|
| `TESTING_HISTORY_DAYS` | clickhouse_mod.rs | history pulled for testing (5 years) |
| `TC_SEARCH_LO/HI_FRAC` | argmin_mod.rs | `tc` search range as a fraction of window length |
| `TC_FILTER_LO/HI_FRAC`, `FIT_R2_MIN` | lib.rs | qualification filter |
| `BREAK_RATIO_HI/LO`, `BREAK_MIN_PRICE` | lib.rs | data-break screen |
| `STOP_LOSS_PCT`, `TC_EXIT_BUFFER_DAYS`, `EXIT_CONF_BARS`, `MAX_HOLD_BARS` | backtester.rs | exit rules |
| production `(p1, p2)` per group | lib.rs, `run_backtests` | live thresholds |

## Diagnostics (`examples/`)

```
cargo run --release --example check_prices data/testing/Micro1.csv   # what the data screen trims
cargo run --release --example trade_counts output/testing            # ungated signal/trade/exit counts
cargo run --release --example dump_stats output/testing stats.csv    # per-ticker, per-strategy aggregates
cargo run --release --example calibrate_r2 production                # R^2 distribution of the fits
cargo run --release --example dotcom                                 # sanity check on the NASDAQ 2000 peak
```

`dump_stats` is the tool for checking that a strategy's edge is not a few
outlier trades: pool across universes and cap gains per trade before
trusting a cell.

## Tests

```
cargo test --release
```

## Layout

- `src/main.rs`: CLI, orchestration, file cleanup
- `src/clickhouse_mod.rs`: price pulls and score inserts
- `src/argmin_mod.rs`: the LPPLS fitter
- `src/lib.rs`: nested fits, qualification, indicators, run helpers, data screen, production thresholds
- `src/backtester.rs`: signals, trade simulation, pooled summary, group files
