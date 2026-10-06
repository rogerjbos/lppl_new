use polars::prelude::*;
use serde::Serialize;
use std::{
    collections::HashSet, env, error::Error as StdError, fmt, fmt::Debug, fs::File, io::Cursor,
    path::Path, result::Result, sync::Arc,
};
use tokio::{fs, task::JoinError};

use crate::clickhouse_mod::write_price_file;

/// Maximum holding period for a backtest position, in bars (~6 trading
/// months). A position not closed by an opposite signal exits at the open
/// this many bars after entry.
pub const MAX_HOLD_BARS: usize = 126;

/// Stop loss in percent, checked at each bar's open against the entry price.
pub const STOP_LOSS_PCT: f64 = 20.0;

/// Calendar days past the predicted critical time tc (median tc of the
/// qualified fits at entry) before a position is closed. tc is the most
/// probable end of the bubble, so the position is given this long after it
/// for the move to play out.
pub const TC_EXIT_BUFFER_DAYS: f64 = 30.0;

/// Close a position once the entry-side confidence has been at or below
/// EXIT_CONF_LEVEL for this many consecutive bars: the residual excursion
/// that justified the trade has reverted. Applies only to trades without a
/// tc target. LPPLS confidence is intermittent even inside a real bubble, so
/// for tc-bearing trades this rule fired within days of entry on nearly
/// every trade and the tc exit never got a chance; those trades run to tc +
/// TC_EXIT_BUFFER_DAYS, the stop, or MAX_HOLD_BARS instead.
pub const EXIT_CONF_BARS: usize = 5;
pub const EXIT_CONF_LEVEL: f64 = 0.0;

#[derive(Clone, Debug, Serialize)]
pub struct Backtest {
    pub ticker: String,
    pub universe: String,
    pub strategy: String,
    pub expectancy: f64,
    pub profit_factor: f64,
    pub hit_ratio: f64,
    pub realized_risk_reward: f64,
    pub avg_gain: f64,
    pub avg_loss: f64,
    pub max_gain: f64,
    pub max_loss: f64,
    pub buys: i32,
    pub sells: i32,
    pub trades: i32,
    // Per-ticker trade sums so summary_performance can pool trades across
    // tickers instead of averaging per-ticker ratios.
    pub wins: i32,
    pub losses: i32,
    pub gain_sum: f64,
    pub loss_sum: f64,
    pub ret_sq_sum: f64,
    // Why trades closed, for tuning the exit rules.
    pub exit_signal: i32,
    pub exit_stop: i32,
    pub exit_tc: i32,
    pub exit_conf: i32,
    pub exit_hold: i32,
    pub exit_eod: i32,
    pub date: String,
    pub buy: i32,
    pub sell: i32,
    pub pos_conf: f32,
    pub neg_conf: f32,
}

#[derive(Debug, Serialize)]
pub struct BuySell {
    pub buy: Vec<i32>,
    pub sell: Vec<i32>,
    pub pos_conf: Vec<f32>,
    pub neg_conf: Vec<f32>,
    /// Predicted critical time (same units as the `time` column) attached
    /// to the signal at that bar; NaN when the signal has none.
    pub exit_time: Vec<f64>,
}

/// Optional f64 column accessor for inputs a signal may not carry.
fn opt_f64(df: &DataFrame, name: &str) -> Option<Float64Chunked> {
    df.column(name).ok().and_then(|c| c.f64().ok().cloned())
}

pub fn test() -> Result<(), Box<dyn StdError>> {
    println!("hello world!");
    Ok(())
}

// #[derive(Debug)]
pub struct Signal {
    pub name: String,
    pub param1: f64,
    pub param2: f64,
    pub f: Arc<dyn Fn(DataFrame) -> BuySell + Send + Sync>,
}

impl fmt::Debug for Signal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Signal")
            .field("name", &self.name)
            .field("param1", &self.param1)
            .field("param2", &self.param2)
            // Don't include the function field in the debug output
            .finish()
    }
}

// Define the function type for your signals
pub type SignalFunction = fn(DataFrame, f64, f64) -> BuySell;

pub fn signal_fun(df: DataFrame, pos_level: f64, neg_level: f64) -> BuySell {
    let len = df.height();
    let mut buy = vec![0; len];
    let mut sell = vec![0; len];
    let mut pos_conf = vec![0.; len];
    let mut neg_conf = vec![0.; len];
    let mut exit_time = vec![f64::NAN; len];

    let pos = df.column("pos_conf").unwrap().f64().unwrap();
    let neg = df.column("neg_conf").unwrap().f64().unwrap();
    let pos_tc = opt_f64(&df, "pos_tc");
    let neg_tc = opt_f64(&df, "neg_tc");
    let tc_at = |col: &Option<Float64Chunked>, i: usize| -> f64 {
        col.as_ref().and_then(|c| c.get(i)).unwrap_or(f64::NAN)
    };

    for i in 1..len - 1 {
        // The final bar's signal uses same-day confidence (production list).
        let src = if i == len - 2 { i + 1 } else { i };
        let pos_0 = pos.get(src).unwrap();
        let neg_0 = neg.get(src).unwrap();

        if pos_0 > pos_level {
            sell[i + 1] = -1;
            exit_time[i + 1] = tc_at(&pos_tc, src);
        } else if neg_0 > neg_level {
            buy[i + 1] = 1;
            exit_time[i + 1] = tc_at(&neg_tc, src);
        }
        pos_conf[i + 1] = pos_0 as f32;
        neg_conf[i + 1] = neg_0 as f32;
    }
    BuySell {
        buy,
        sell,
        pos_conf,
        neg_conf,
        exit_time,
    }
}

// Normalized-residual signal (arXiv:2510.10878): contrarian entries when
// eps_norm (a residual z-score, see compute_residual_indicator) has
// exceeded +/-tau for dmin consecutive days. Same one-bar
// delay convention as signal_fun: the signal at bar i+1 uses data through
// bar i, and trades execute at the open of the signal bar.
pub fn res_signal_fun(df: DataFrame, tau: f64, dmin: usize) -> BuySell {
    let len = df.height();
    let mut buy = vec![0; len];
    let mut sell = vec![0; len];
    let mut pos_conf = vec![0.; len];
    let mut neg_conf = vec![0.; len];

    let eps = df.column("eps_norm").unwrap().f64().unwrap();

    let mut pos_run = 0usize;
    let mut neg_run = 0usize;
    for i in 0..len.saturating_sub(1) {
        let mut e = eps.get(i).unwrap_or(0.0);
        if e.is_nan() {
            e = 0.0;
        }
        if e >= tau {
            pos_run += 1;
        } else {
            pos_run = 0;
        }
        if e <= -tau {
            neg_run += 1;
        } else {
            neg_run = 0;
        }
        if pos_run >= dmin {
            // price far above the fitted bubble trajectory -> overheated
            sell[i + 1] = -1;
        } else if neg_run >= dmin {
            buy[i + 1] = 1;
        }
        pos_conf[i + 1] = e.max(0.0) as f32;
        neg_conf[i + 1] = (-e).max(0.0) as f32;
    }
    // No tc for residual signals: exits come from the confidence rule
    // (eps_norm reverting to the fitted trajectory), the stop, or max hold.
    BuySell {
        buy,
        sell,
        pos_conf,
        neg_conf,
        exit_time: vec![f64::NAN; len],
    }
}

async fn concat_dataframes(dfs: Vec<DataFrame>) -> Result<DataFrame, PolarsError> {
    if dfs.is_empty() {
        return Err(PolarsError::ComputeError(
            "No DataFrames to concatenate.".into(),
        ));
    }

    // Get the schema of the first DataFrame as the reference schema
    let reference_schema = dfs[0].schema();
    let mut compatible_dfs = Vec::new();
    let mut incompatible_dfs = Vec::new();

    // Separate compatible and incompatible DataFrames
    for df in dfs {
        let schema = df.schema();
        let is_compatible = schema
            .iter_fields()
            .zip(reference_schema.iter_fields())
            .all(|(field, ref_field)| field.dtype() == ref_field.dtype());

        if is_compatible {
            compatible_dfs.push(df);
        } else {
            incompatible_dfs.push((df, schema));
        }
    }

    // For debugging, print details of incompatible DataFrames (up to 3) versus reference
    for (i, (_df, schema)) in incompatible_dfs.iter().enumerate() {
        if i == 0 {
            println!("reference_schema: {:?}", reference_schema);
        }
        if i < 3 {
            println!("Incompatible DataFrame {}: Schema: {:?}", i + 1, schema);
        }
    }

    // Convert compatible DataFrames to LazyFrames for concatenation
    let lazy_frames: Vec<LazyFrame> = compatible_dfs.into_iter().map(|df| df.lazy()).collect();

    // Use the concat function for LazyFrames
    let concatenated_lazy_frame = concat(&lazy_frames, UnionArgs::default())?;

    // Collect the concatenated LazyFrame back into a DataFrame
    let result_df = concatenated_lazy_frame.collect()?;
    Ok(result_df)
}

pub async fn summary_performance_file(
    path: String,
    production: bool,
    univ: Vec<String>,
) -> Result<String, Box<dyn StdError>> {
    println!(
        "Performance starting for {}",
        if production { "Production" } else { "Testing" }
    );

    let bt_col_names = vec![
        "ticker",
        "universe",
        "strategy",
        "expectancy",
        "profit_factor",
        "hit_ratio",
        "realized_risk_reward",
        "avg_gain",
        "avg_loss",
        "max_gain",
        "max_loss",
        "buys",
        "sells",
        "trades",
        "wins",
        "losses",
        "gain_sum",
        "loss_sum",
        "ret_sq_sum",
        "exit_signal",
        "exit_stop",
        "exit_tc",
        "exit_conf",
        "exit_hold",
        "exit_eod",
        "date",
        "buy",
        "sell",
        "pos_conf",
        "neg_conf",
    ];
    let set_bt: HashSet<_> = bt_col_names.iter().cloned().collect();

    let b_names = vec![
        "ticker", "universe", "strategy", "date", "buy", "sell", "pos_conf", "neg_conf",
    ];
    let stocks = !univ.contains(&"Crypto".to_string());

    let folder = match (stocks, production) {
        (true, true) => "output/production",
        (true, false) => "output/testing",
        (false, true) => "output_crypto/production",
        (false, false) => "output_crypto/testing",
    };

    let dir_path = format!("{}/{}", path, folder);
    let mut a: Vec<DataFrame> = Vec::new();
    let mut b: Vec<DataFrame> = Vec::new();
    let mut entries = fs::read_dir(&dir_path).await?;
    // println!("summary performance dir_path: {}", &dir_path);

    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        // println!("path: {:?}", &path);

        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("parquet") {
            // println!("here 1: {:?}", &path.to_str());

            let lf = LazyFrame::scan_parquet(
                path.to_str().expect("path error"),
                ScanArgsParquet::default(),
            )?
            .with_column(col("expectancy").fill_null(lit(0.0)))
            .with_column(col("profit_factor").fill_null(lit(0.0)))
            .with_column(col("hit_ratio").fill_null(lit(0.0)))
            .with_column(col("realized_risk_reward").fill_null(lit(0.0)))
            .with_column(col("avg_gain").fill_null(lit(0.0)))
            .with_column(col("avg_loss").fill_null(lit(0.0)))
            .with_column(col("max_gain").fill_null(lit(0.0)))
            .with_column(col("max_loss").fill_null(lit(0.0)))
            .collect();

            match lf {
                Ok(df) => {
                    // println!("lf: {:?}", df);
                    // println!("lf column names: {:?}", df.get_column_names());

                    // Ensure all required columns are present
                    let df_names = df.get_column_names();
                    let set_df: HashSet<_> = df_names.into_iter().map(|s| s.as_str()).collect();
                    if set_bt.is_subset(&set_df) {
                        a.push(df.select(bt_col_names.clone())?);
                        b.push(df.select(b_names.clone())?);
                    }
                }
                Err(e) => println!("Error processing file {}: {}", path.display(), e),
            }
        }
    }
    // println!("here 7 a: {:?}", a);

    // ALL
    let df = concat_dataframes(a).await?;
    // println!("parquet: {:?}", df.clone());
    // println!("parquet columns: {:?}", df.clone().get_column_names());

    let mut out = summary_performance(df.clone())?;
    let datetag = df
        .column("date")?
        .get(0)?
        .to_string()
        .trim_matches('"')
        .replace("-", "");
    let tag: &str = if stocks { "stocks" } else { "crypto" };

    // show and save performance only for testing
    if !production {
        println!("Average Performance by Strategy:\n {:?}", out);
        // let tag: &str = &univ.join("_");

        let perf_filename = if production {
            format!("{}/performance/{}_all_{}.csv", path, tag, &datetag)
        } else {
            format!("{}/performance/{}_testing.csv", path, tag)
        };

        let mut file = File::create(perf_filename)?;
        let _ = CsvWriter::new(&mut file).finish(&mut out);
    }

    // show coverage
    if production {
        // concat all the price dfs
        let mut p: Vec<DataFrame> = Vec::new();
        // println!("univ: {:?}", univ);

        // let univ = ["Crypto","LC1","LC2","MC1","MC2","SC1","SC2","SC3","SC4","Micro1","Micro2"];
        for u in univ {
            let file_path = format!("{}/data/production/{}.csv", path, u);
            // let tmp = LazyFrame::scan_parquet(file_path, ScanArgsParquet::default())?;
            // println!("file_path: {}", file_path.clone());
            let mut schema = Schema::with_capacity(8);
            schema.with_column("Date".into(), DataType::String);
            schema.with_column("Ticker".into(), DataType::String);
            schema.with_column("Universe".into(), DataType::String);
            schema.with_column("Open".into(), DataType::Float64);
            schema.with_column("High".into(), DataType::Float64);
            schema.with_column("Low".into(), DataType::Float64);
            schema.with_column("Close".into(), DataType::Float64);
            schema.with_column("Volume".into(), DataType::Float64);
            let schema = Arc::new(schema);

            let tmp = LazyCsvReader::new(file_path)
                .with_schema(Some(schema))
                .with_has_header(true)
                .finish()?;

            let tmp = tmp.with_column(
                col("Date")
                    .str()
                    .strptime(
                        DataType::Date, // First argument: desired output data type
                        StrptimeOptions {
                            format: Some("%Y-%m-%d".into()),
                            strict: false,
                            exact: true,
                            cache: true,
                        },
                        lit(NULL), // Third argument: handling for ambiguous cases, using null as default
                    )
                    .alias("Date"),
            );
            // println!("tmp: {:?}", tmp.clone().collect());

            let grouped = tmp
                .group_by_stable([col("Ticker")])
                .agg([
                    col("Date").count().alias("observations"),
                    col("Date").last().alias("last date"),
                ])
                .sort(
                    vec!["Ticker"],
                    SortMultipleOptions {
                        descending: vec![false],
                        nulls_last: vec![true],
                        ..Default::default()
                    },
                );
            // println!("grouped: {:?}", grouped.clone().collect()?);

            p.push(grouped.collect().unwrap());
        }
        let all_p = concat_dataframes(p).await?;

        let df_grouped = df
            .lazy()
            .group_by_stable([col("ticker")])
            .agg([col("strategy").count().alias("strategies")])
            .sort(
                vec!["ticker"],
                SortMultipleOptions {
                    descending: vec![false],
                    nulls_last: vec![true],
                    ..Default::default()
                },
            );

        let both = all_p
            .lazy()
            .inner_join(df_grouped, col("Ticker"), col("ticker"))
            // .filter(
            //     col("strategies").lt(lit(121))
            // )
            .sort(
                vec!["strategies"],
                SortMultipleOptions {
                    descending: vec![false],
                    ..Default::default()
                },
            )
            .collect();
        println!("Strategy Coverage: {:?}", both);

        // buys and sells for the current date
        let df_b = concat_dataframes(b).await?;
        let mut buys = df_b
            .clone()
            .lazy()
            .filter(col("buy").eq(lit(1)))
            .sort(
                vec!["ticker"],
                SortMultipleOptions {
                    descending: vec![false],
                    ..Default::default()
                },
            )
            .collect()?;

        let mut sells = df_b
            .lazy()
            .filter(col("sell").eq(lit(-1)))
            .sort(
                vec!["ticker"],
                SortMultipleOptions {
                    descending: vec![false],
                    ..Default::default()
                },
            )
            .collect()?;

        let buy_filename = format!("{}/performance/{}_buys_{}.csv", path, tag, datetag);
        let mut buy_file = File::create(buy_filename)?;
        let _ = CsvWriter::new(&mut buy_file).finish(&mut buys);

        let sell_filename = format!("{}/performance/{}_sells_{}.csv", path, tag, datetag);
        let mut sell_file = File::create(sell_filename)?;
        let _ = CsvWriter::new(&mut sell_file).finish(&mut sells);
    };

    // only show for testing
    if !production {
        // LC
        let lc = out
            .clone()
            .lazy()
            .filter(
                col("universe")
                    .eq(lit("LC1"))
                    .or(col("universe").eq(lit("LC2"))),
            )
            .collect();

        match lc {
            Ok(ref _df) => {
                let perf_filename = format!("{}/performance/{}.csv", path, "LC");
                let mut file = File::create(perf_filename)?;
                let _ = CsvWriter::new(&mut file).finish(&mut lc?);
            }
            Err(ref e) => println!("Error filtering DataFrame for LC: \n{:?}", e),
        }

        // MC
        let mc = out
            .clone()
            .lazy()
            .filter(
                col("universe")
                    .eq(lit("MC1"))
                    .or(col("universe").eq(lit("MC2"))),
            )
            .collect();

        match mc {
            Ok(ref _df) => {
                let perf_filename = format!("{}/performance/{}.csv", path, "MC");
                let mut file = File::create(perf_filename)?;
                let _ = CsvWriter::new(&mut file).finish(&mut mc?);
            }
            Err(ref e) => println!("Error filtering DataFrame for MC: \n{:?}", e),
        }

        // SC
        let sc = out
            .clone()
            .lazy()
            .filter(
                col("universe")
                    .eq(lit("SC1"))
                    .or(col("universe").eq(lit("SC2")))
                    .or(col("universe").eq(lit("SC3")))
                    .or(col("universe").eq(lit("SC4"))),
            )
            .collect();

        match sc {
            Ok(ref _df) => {
                let perf_filename = format!("{}/performance/{}.csv", path, "SC");
                let mut file = File::create(perf_filename)?;
                let _ = CsvWriter::new(&mut file).finish(&mut sc?);
            }
            Err(ref e) => println!("Error filtering DataFrame for SC: \n{:?}", e),
        }

        // Microcap
        let micro = out
            .lazy()
            .filter(
                col("universe")
                    .eq(lit("Micro1"))
                    .or(col("universe").eq(lit("Micro2"))),
            )
            .collect();

        match micro {
            Ok(ref _df) => {
                let perf_filename = format!("{}/performance/{}.csv", path, "Micro");
                let mut file = File::create(perf_filename)?;
                let _ = CsvWriter::new(&mut file).finish(&mut micro?);
            }
            Err(ref e) => println!("Error filtering DataFrame for Micro: \n{:?}", e),
        }
    }

    Ok(datetag)
}

// Pool all trades of a (strategy, universe) across tickers and compute the
// metrics on the pool. Averaging per-ticker ratios instead gave each ticker
// equal weight regardless of trade count, so tickers with a single trade
// (profit_factor 999 or 0, expectancy +/-2000) dominated the summary.
pub fn summary_performance(df: DataFrame) -> Result<DataFrame, Box<dyn StdError>> {
    let f64_ = |name: &str| col(name).cast(DataType::Float64);
    // population variance of the per-trade returns (clamped at 0 below)
    let variance = col("ret_sq_sum") / f64_("total_trades") - col("expectancy").pow(lit(2.0));

    let out = df
        .lazy()
        .group_by_stable([col("strategy"), col("universe")])
        .agg([
            col("ticker").count().alias("N"),
            col("trades").gt(lit(0)).sum().alias("tickers_traded"),
            col("trades").sum().alias("total_trades"),
            col("wins").sum().alias("wins"),
            col("losses").sum().alias("losses"),
            col("gain_sum").sum().alias("gain_sum"),
            col("loss_sum").sum().alias("loss_sum"),
            col("ret_sq_sum").sum().alias("ret_sq_sum"),
            col("exit_signal").sum().alias("exit_signal"),
            col("exit_stop").sum().alias("exit_stop"),
            col("exit_tc").sum().alias("exit_tc"),
            col("exit_conf").sum().alias("exit_conf"),
            col("exit_hold").sum().alias("exit_hold"),
            col("exit_eod").sum().alias("exit_eod"),
            col("max_gain").max().alias("max_gain"),
            col("max_loss").min().alias("max_loss"),
            col("buys").mean().alias("buys"),
            col("sells").mean().alias("sells"),
            col("trades").mean().alias("trades"),
        ])
        // LPPLS signals are rare by design, so gate on total trades across
        // the universe (statistical validity), not mean trades per ticker.
        .filter(col("total_trades").gt_eq(lit(20)))
        .with_columns([
            (f64_("wins") / f64_("total_trades") * lit(100.0)).alias("hit_ratio"),
            when(col("wins").gt(lit(0)))
                .then(col("gain_sum") / f64_("wins"))
                .otherwise(lit(0.0))
                .alias("avg_gain"),
            when(col("losses").gt(lit(0)))
                .then(col("loss_sum") / f64_("losses"))
                .otherwise(lit(0.0))
                .alias("avg_loss"),
            // mean percent return per trade
            ((col("gain_sum") - col("loss_sum")) / f64_("total_trades")).alias("expectancy"),
        ])
        .with_columns([
            when(col("loss_sum").gt(lit(0.0)))
                .then(col("gain_sum") / col("loss_sum"))
                .otherwise(lit(999.0))
                .alias("profit_factor"),
            when(col("avg_loss").gt(lit(0.0)))
                .then(col("avg_gain") / col("avg_loss"))
                .otherwise(lit(999.0))
                .alias("risk_reward"),
            // standard error of the mean return per trade
            (when(variance.clone().gt(lit(0.0)))
                .then(variance.clone())
                .otherwise(lit(0.0))
                .sqrt()
                / f64_("total_trades").sqrt())
            .alias("std_err"),
        ])
        .with_column(
            when(col("std_err").gt(lit(0.0)))
                .then(col("expectancy") / col("std_err"))
                .otherwise(lit(0.0))
                .alias("t_stat"),
        )
        .select([
            col("strategy"),
            col("universe"),
            col("hit_ratio"),
            col("risk_reward"),
            col("avg_gain"),
            col("avg_loss"),
            col("max_gain"),
            col("max_loss"),
            col("buys"),
            col("sells"),
            col("trades"),
            col("total_trades"),
            col("tickers_traded"),
            col("N"),
            col("expectancy"),
            col("std_err"),
            col("t_stat"),
            col("profit_factor"),
            col("exit_signal"),
            col("exit_stop"),
            col("exit_tc"),
            col("exit_conf"),
            col("exit_hold"),
            col("exit_eod"),
        ])
        .sort(
            vec!["t_stat"],
            SortMultipleOptions {
                descending: vec![false],
                ..Default::default()
            },
        )
        .collect()?;

    Ok(out)
}

// Apply a signal function to data and calculate strategy performance
pub async fn sig(df: LazyFrame, signal: &Signal) -> Result<Backtest, Box<dyn StdError>> {
    let func = &signal.f;
    let s = func(df.clone().collect()?);
    let bt = backtest_performance(df.collect()?, s, &signal.name)?;
    // println!("bt_sig: {:?}", bt.clone());
    Ok(bt)
}

pub async fn run_all_backtests(
    df: LazyFrame,
    signals: Vec<Signal>,
) -> Result<Vec<Backtest>, JoinError> {
    // wrap df in an Arc for shared ownership across tasks
    let df = Arc::new(df);

    let futures: Vec<_> = signals
        .into_iter()
        .map(|signal| {
            // clone Arc for each task
            let df_clone = Arc::clone(&df);
            tokio::spawn(async move { sig(df_clone.as_ref().clone(), &signal).await.unwrap() })
        })
        .collect();

    let results = futures::future::join_all(futures).await;

    // Handle the results, assuming `sig` returns `Result<Backtest, _>`
    let backtests: Vec<Backtest> = results.into_iter().filter_map(Result::ok).collect();

    // let _ = showbt(backtests[0].clone());
    Ok(backtests)
}

pub async fn create_price_files(
    univ_vec: Vec<String>,
    production: bool,
) -> Result<(), Box<dyn StdError>> {
    let folder = if production { "production" } else { "testing" };

    for u in univ_vec {
        let user_path = match env::var("CLICKHOUSE_USER_PATH") {
            Ok(path) => path,
            Err(_) => String::from("/srv"),
        };
        let file_path = format!(
            "{}/rust_home/lppl_new/data/{}/{}.csv",
            user_path,
            folder.to_string(),
            u.to_string()
        );
        let file_path: &str = &file_path;
        if production == false && Path::new(&file_path).exists() {
            println!("Price file exists for {}", file_path);
        } else {
            println!("Price file generating for {}", file_path);
            write_price_file(u, production).await?;
        }
    }
    Ok(())
}

pub fn backtest_performance(
    df: DataFrame,
    side: BuySell,
    strategy: &str,
) -> Result<Backtest, Box<dyn StdError>> {
    let len = df.height();

    let open = df.column("Open").unwrap().f64().unwrap();
    let close = df.column("Close").unwrap().f64().unwrap();

    // Position-based trade simulation with percent returns (comparable
    // across tickers regardless of price level).
    // - A buy signal opens a long, a sell signal opens a short.
    // - Repeated same-direction signals while a position is open are
    //   ignored (no chaining into 1-bar trades).
    // - Exits are checked at each bar's open, in this priority:
    //     1. an opposite signal closes and reverses into the new direction;
    //     2. stop loss: open-to-entry return <= -STOP_LOSS_PCT;
    //     3. tc exit: the bar's time is past the signal's predicted
    //        critical time plus TC_EXIT_BUFFER_DAYS;
    //     4. confidence exit (trades with no tc target only): entry-side
    //        confidence at or below EXIT_CONF_LEVEL for EXIT_CONF_BARS
    //        consecutive bars;
    //     5. max hold: MAX_HOLD_BARS bars since entry.
    //   Any position still open at the end of the data is marked to market
    //   at the final close.
    // - The final bar's signal is derived from same-day confidence (kept
    //   for the production buy/sell list), so the backtest ignores it.
    let time = opt_f64(&df, "time");
    let pct = |position: i32, entry: f64, exit: f64| -> f64 {
        if position == 1 {
            (exit / entry - 1.0) * 100.0
        } else {
            (entry - exit) / entry * 100.0
        }
    };

    let mut trade_results: Vec<f64> = Vec::new();
    let mut position: i32 = 0; // 0 = flat, 1 = long, -1 = short
    let mut entry_price = f64::NAN;
    let mut entry_bar: usize = 0;
    let mut tc_target = f64::NAN;
    let mut conf_gone_run: usize = 0;
    let (mut exit_signal, mut exit_stop, mut exit_tc, mut exit_conf, mut exit_hold, mut exit_eod) =
        (0, 0, 0, 0, 0, 0);

    for i in 0..len {
        let is_last = i == len - 1;
        let buy_sig = !is_last && side.buy[i] == 1;
        let sell_sig = !is_last && side.sell[i] == -1;

        if position != 0 {
            let open_i = open.get(i).unwrap_or(f64::NAN);
            // Entry-side confidence at this bar (data through the prior close).
            let conf = if position == 1 {
                side.neg_conf.get(i)
            } else {
                side.pos_conf.get(i)
            }
            .copied()
            .unwrap_or(0.0) as f64;
            if conf <= EXIT_CONF_LEVEL {
                conf_gone_run += 1;
            } else {
                conf_gone_run = 0;
            }

            let opposite = (position == 1 && sell_sig) || (position == -1 && buy_sig);
            let open_ret = pct(position, entry_price, open_i);
            let stopped = open_ret.is_finite() && open_ret <= -STOP_LOSS_PCT;
            let past_tc = tc_target.is_finite()
                && time
                    .as_ref()
                    .and_then(|t| t.get(i))
                    .map_or(false, |ti| ti > tc_target + TC_EXIT_BUFFER_DAYS);
            let conf_gone = !tc_target.is_finite() && conf_gone_run >= EXIT_CONF_BARS;
            let expired = i - entry_bar >= MAX_HOLD_BARS;

            let counter: Option<&mut i32> = if is_last {
                Some(&mut exit_eod)
            } else if opposite {
                Some(&mut exit_signal)
            } else if stopped {
                Some(&mut exit_stop)
            } else if past_tc {
                Some(&mut exit_tc)
            } else if conf_gone {
                Some(&mut exit_conf)
            } else if expired {
                Some(&mut exit_hold)
            } else {
                None
            };
            if let Some(counter) = counter {
                let exit_price = if is_last {
                    close.get(i).unwrap_or(f64::NAN)
                } else {
                    open_i
                };
                if entry_price > 0.0 && exit_price.is_finite() {
                    trade_results.push(pct(position, entry_price, exit_price));
                }
                *counter += 1;
                position = 0;
                entry_price = f64::NAN;
                tc_target = f64::NAN;
            }
        }

        if position == 0 && (buy_sig || sell_sig) {
            let price = open.get(i).unwrap_or(f64::NAN);
            if price > 0.0 {
                position = if buy_sig { 1 } else { -1 };
                entry_price = price;
                entry_bar = i;
                tc_target = side.exit_time.get(i).copied().unwrap_or(f64::NAN);
                conf_gone_run = 0;
            }
        }
    }

    // Profit factor
    let total_net_profits: Vec<f64> = trade_results
        .iter()
        .copied()
        .filter(|&x| x > 0.0)
        .collect();
    let total_net_losses: Vec<f64> = trade_results
        .iter()
        .copied()
        .filter(|&x| x < 0.0)
        .collect();
    let sum_total_net_profits = total_net_profits.iter().sum::<f64>();
    let sum_total_net_losses = total_net_losses.iter().sum::<f64>().abs();
    let profit_factor: f64 = {
        let pf = sum_total_net_profits / sum_total_net_losses;
        if pf.is_nan() {
            0.0
        } else {
            f64::min(999.0, pf)
        }
    };

    // Hit ratio
    let hit_ratio: f64 = {
        let hr = (total_net_profits.len() as f64
            / (total_net_losses.len() + total_net_profits.len()) as f64)
            * 100.0;
        if hr.is_nan() {
            0.0
        } else {
            f64::min(100., hr)
        }
    };

    // Risk reward ratio
    let average_gain: f64 = {
        let ag = sum_total_net_profits / total_net_profits.len() as f64;
        if ag.is_nan() {
            0.0
        } else {
            ag
        }
    };
    let average_loss: f64 = {
        let al = sum_total_net_losses / total_net_losses.len() as f64;
        if al.is_nan() {
            0.0
        } else {
            al
        }
    };
    let realized_risk_reward: f64 = {
        let rr = average_gain / average_loss;
        if rr.is_nan() {
            0.0
        } else {
            f64::min(999., rr)
        }
    };
    let trades: i32 = trade_results.len() as i32;
    let wins = total_net_profits.len() as i32;
    let losses = total_net_losses.len() as i32;
    let ret_sq_sum: f64 = trade_results.iter().map(|r| r * r).sum();

    // Expectancy: mean percent return per trade
    let expectancy = if trades > 0 {
        (sum_total_net_profits - sum_total_net_losses) / trades as f64
    } else {
        0.0
    };

    let max_gain = total_net_profits
        .into_iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap());
    let max_loss = total_net_losses
        .into_iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap());

    let buys = side.buy.iter().sum::<i32>();
    let sells = side.sell.iter().sum::<i32>().abs();

    let buy = side.buy[len - 1];
    let sell = side.sell[len - 1];
    let pos_conf = {
        if let Some(pos) = side.pos_conf.get(len - 1) {
            if pos.is_nan() {
                0.0
            } else {
                *pos
            }
        } else {
            0.0 // Fallback if pos_conf is Null
        }
    };
    let neg_conf = {
        if let Some(neg) = side.neg_conf.get(len - 1) {
            if neg.is_nan() {
                0.0
            } else {
                *neg
            }
        } else {
            0.0 // Fallback if pos_conf is Null
        }
    };
    let quoted_ticker = df.column("Ticker").unwrap().get(0).unwrap().to_string();
    let ticker = quoted_ticker.trim_matches('"').to_string();
    let universe1 = df.column("Universe").unwrap().get(0).unwrap().to_string();
    let universe = universe1.trim_matches('"').to_string();
    let quoted_date = df.column("Date").unwrap().get(len - 1).unwrap().to_string();
    let date = quoted_date.trim_matches('"').to_string();
    // println!("finished {} signal {:?}", ticker, strategy);

    Ok(Backtest {
        ticker: ticker,
        universe: universe,
        strategy: strategy.to_string(),
        expectancy,
        profit_factor: profit_factor,
        hit_ratio: hit_ratio,
        realized_risk_reward: realized_risk_reward,
        avg_gain: average_gain,
        avg_loss: average_loss,
        max_gain: match max_gain {
            Some(x) => x,
            None => 0.0,
        },
        max_loss: match max_loss {
            Some(x) => x,
            None => 0.0,
        },
        buys: buys,
        sells: sells,
        trades: trades,
        wins,
        losses,
        gain_sum: sum_total_net_profits,
        loss_sum: sum_total_net_losses,
        ret_sq_sum,
        exit_signal,
        exit_stop,
        exit_tc,
        exit_conf,
        exit_hold,
        exit_eod,
        date: date,
        buy: buy,
        sell: sell,
        pos_conf: pos_conf,
        neg_conf: neg_conf,
    })
}

pub fn showbt(bt: Backtest) -> Result<(), Box<dyn StdError>> {
    println!("");
    println!("Ticker:           {}", bt.ticker);
    println!("Universe:         {}", bt.universe);
    println!("Strategy:         {}", bt.strategy);
    println!("Profit Factor:    {:.1}", bt.profit_factor);
    println!("Hit Ratio:        {:.1}", bt.hit_ratio);
    println!("Expectancy:       {:.1}", bt.expectancy);
    println!("Risk-Reward:      {:.1}", bt.realized_risk_reward);
    println!("Avg Gain:         {:.1}", bt.avg_gain);
    println!("Avg Loss:         {:.1}", bt.avg_loss);
    println!("Max Gain:         {:.1}", bt.max_gain);
    println!("Max Loss:         {:.1}", bt.max_loss);
    println!("Buys:             {:.1}", bt.buys);
    println!("Sells:            {:.1}", bt.sells);
    println!("Trades:           {:.1}", bt.trades);
    println!("Pos Conf:         {:.1}", bt.pos_conf);
    println!("Neg Conf:         {:.1}", bt.neg_conf);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtest_performance_percent_returns() {
        let df = polars::df!(
            "Date" => &["d1", "d2", "d3", "d4", "d5", "d6"],
            "Ticker" => &["TST"; 6],
            "Universe" => &["U"; 6],
            "Open" => &[100.0, 100.0, 110.0, 120.0, 115.0, 95.0],
            "Close" => &[100.0, 105.0, 115.0, 118.0, 110.0, 90.0],
        )
        .unwrap();

        // Long entered at bar 1 reverses into a short at the sell signal
        // (bar 3); the short is still open at the end and is marked to
        // market at the final close. The final bar's sell signal is
        // ignored by the backtest (same-day confidence).
        let side = BuySell {
            buy: vec![0, 1, 0, 0, 0, 0],
            sell: vec![0, 0, 0, -1, 0, -1],
            pos_conf: vec![0.0; 6],
            neg_conf: vec![0.0; 6],
            exit_time: vec![f64::NAN; 6],
        };

        let bt = backtest_performance(df, side, "test").unwrap();

        // long: 100 -> 120 = +20%, short: 120 -> 90 (last close) = +25%
        assert_eq!(bt.trades, 2);
        assert!((bt.avg_gain - 22.5).abs() < 1e-9, "avg_gain: {}", bt.avg_gain);
        assert!((bt.max_gain - 25.0).abs() < 1e-9, "max_gain: {}", bt.max_gain);
        assert!((bt.hit_ratio - 100.0).abs() < 1e-9);
        assert_eq!(bt.buys, 1);
        assert_eq!(bt.sells, 2);
        // pooling sums: 2 wins, +45% total, no losses, mean +22.5%/trade
        assert_eq!(bt.wins, 2);
        assert_eq!(bt.losses, 0);
        assert!((bt.gain_sum - 45.0).abs() < 1e-9);
        assert!((bt.loss_sum - 0.0).abs() < 1e-9);
        assert!((bt.ret_sq_sum - (400.0 + 625.0)).abs() < 1e-9);
        assert!((bt.expectancy - 22.5).abs() < 1e-9);
    }

    #[test]
    fn test_summary_performance_pools_trades() {
        // Ticker A: 3 trades (+10, +20, -10). Ticker B: 1 trade (-30).
        // Equal-weighted per-ticker means would give hit_ratio 33% and a
        // profit factor dominated by B's 0; pooled: 2 wins of 4 = 50%,
        // gains 30 vs losses 40, mean per trade -2.5%.
        let mk = |ticker: &str, trades: i32, wins: i32, gain: f64, loss: f64, sq: f64| Backtest {
            ticker: ticker.to_string(),
            universe: "U".to_string(),
            strategy: "s".to_string(),
            expectancy: 0.0,
            profit_factor: 0.0,
            hit_ratio: 0.0,
            realized_risk_reward: 0.0,
            avg_gain: 0.0,
            avg_loss: 0.0,
            max_gain: gain,
            max_loss: -loss,
            buys: trades,
            sells: 0,
            trades,
            wins,
            losses: trades - wins,
            gain_sum: gain,
            loss_sum: loss,
            ret_sq_sum: sq,
            exit_signal: 0,
            exit_stop: 0,
            exit_tc: 0,
            exit_conf: 0,
            exit_hold: trades,
            exit_eod: 0,
            date: "d".to_string(),
            buy: 0,
            sell: 0,
            pos_conf: 0.0,
            neg_conf: 0.0,
        };
        // 20 copies of A-like tickers plus B to clear the 20-trade gate:
        // 20 * (3 trades: 2 wins +30, 1 loss 10) + B (1 loss 30)
        let mut rows: Vec<Backtest> = (0..20)
            .map(|i| mk(&format!("A{}", i), 3, 2, 30.0, 10.0, 100.0 + 400.0 + 100.0))
            .collect();
        rows.push(mk("B", 1, 0, 0.0, 30.0, 900.0));

        let json = serde_json::to_string(&rows).unwrap();
        let df = JsonReader::new(Cursor::new(json)).finish().unwrap();
        let out = summary_performance(df).unwrap();
        assert_eq!(out.height(), 1);

        let get = |c: &str| out.column(c).unwrap().get(0).unwrap().try_extract::<f64>().unwrap();
        let total = 61.0;
        let wins = 40.0;
        assert!((get("hit_ratio") - wins / total * 100.0).abs() < 1e-9);
        assert!((get("avg_gain") - 600.0 / 40.0).abs() < 1e-9);
        assert!((get("avg_loss") - 230.0 / 21.0).abs() < 1e-9);
        assert!((get("expectancy") - (600.0 - 230.0) / total).abs() < 1e-9);
        assert!((get("profit_factor") - 600.0 / 230.0).abs() < 1e-9);
        assert!(get("std_err") > 0.0);
    }

    #[test]
    fn test_repeated_buy_signals_do_not_chain() {
        let df = polars::df!(
            "Date" => &["d1", "d2", "d3", "d4", "d5", "d6"],
            "Ticker" => &["TST"; 6],
            "Universe" => &["U"; 6],
            "Open" => &[100.0, 100.0, 105.0, 110.0, 120.0, 125.0],
            "Close" => &[100.0, 102.0, 107.0, 112.0, 122.0, 130.0],
        )
        .unwrap();

        // Consecutive buy signals used to chain into 1-bar trades; now
        // they hold a single long, marked to market at the final close.
        let side = BuySell {
            buy: vec![0, 1, 1, 1, 0, 0],
            sell: vec![0, 0, 0, 0, 0, 0],
            pos_conf: vec![0.0; 6],
            neg_conf: vec![0.0; 6],
            exit_time: vec![f64::NAN; 6],
        };

        let bt = backtest_performance(df, side, "test").unwrap();

        // one long: 100 -> 130 (last close) = +30%
        assert_eq!(bt.trades, 1);
        assert!((bt.max_gain - 30.0).abs() < 1e-9, "max_gain: {}", bt.max_gain);
    }

    #[test]
    fn test_res_signal_fun_sustained_threshold() {
        let eps = vec![0.0, 0.9, 0.9, 0.9, 0.9, -0.9, 0.0, 0.0];
        let df = polars::df!("eps_norm" => eps).unwrap();

        // tau 0.8, 3-day minimum: the positive run reaches 3 at bar 3, so
        // sell signals fire at bars 4 and 5. The single -0.9 day never
        // reaches the 3-day minimum, so no buy fires.
        let s = res_signal_fun(df, 0.8, 3);
        assert_eq!(s.sell, vec![0, 0, 0, 0, -1, -1, 0, 0]);
        assert_eq!(s.buy, vec![0; 8]);
        assert!((s.pos_conf[2] - 0.9).abs() < 1e-6);
        assert!((s.neg_conf[6] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_max_holding_period_exit() {
        let n = MAX_HOLD_BARS + 10;
        let dates: Vec<String> = (0..n).map(|i| format!("d{}", i)).collect();
        let opens: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
        let closes = opens.clone();
        let mut buy = vec![0; n];
        buy[1] = 1;

        let df = polars::df!(
            "Date" => dates,
            "Ticker" => vec!["TST"; n],
            "Universe" => vec!["U"; n],
            "Open" => opens,
            "Close" => closes,
        )
        .unwrap();

        // Confidence stays up so the confidence exit does not fire first.
        let side = BuySell {
            buy,
            sell: vec![0; n],
            pos_conf: vec![0.0; n],
            neg_conf: vec![0.5; n],
            exit_time: vec![f64::NAN; n],
        };

        let bt = backtest_performance(df, side, "test").unwrap();

        // Entered at open[1] = 101, expired MAX_HOLD_BARS later at
        // open[1 + MAX_HOLD_BARS] = 101 + 126 = 227.
        let expected = (227.0 / 101.0 - 1.0) * 100.0;
        assert_eq!(bt.trades, 1);
        assert_eq!(bt.exit_hold, 1);
        assert!(
            (bt.max_gain - expected).abs() < 1e-9,
            "max_gain: {} expected: {}",
            bt.max_gain,
            expected
        );
    }

    /// Rising price series with a `time` column of consecutive days, a
    /// single buy at bar 1, and constant entry-side confidence.
    fn long_fixture(n: usize, neg_conf: f32) -> (DataFrame, BuySell) {
        let dates: Vec<String> = (0..n).map(|i| format!("d{}", i)).collect();
        let opens: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
        let time: Vec<f64> = (0..n).map(|i| 1000.0 + i as f64).collect();
        let df = polars::df!(
            "Date" => dates,
            "Ticker" => vec!["TST"; n],
            "Universe" => vec!["U"; n],
            "Open" => opens.clone(),
            "Close" => opens,
            "time" => time,
        )
        .unwrap();
        let mut buy = vec![0; n];
        buy[1] = 1;
        let side = BuySell {
            buy,
            sell: vec![0; n],
            pos_conf: vec![0.0; n],
            neg_conf: vec![neg_conf; n],
            exit_time: vec![f64::NAN; n],
        };
        (df, side)
    }

    #[test]
    fn test_tc_exit() {
        let n = 60;
        let (df, mut side) = long_fixture(n, 0.5);
        // predicted tc two days after entry (time 1001): exit at the first
        // bar past 1003 + 30 = 1033, i.e. bar 34 (time 1034), open 134.
        side.exit_time[1] = 1003.0;

        let bt = backtest_performance(df, side, "test").unwrap();
        let expected = (134.0 / 101.0 - 1.0) * 100.0;
        assert_eq!(bt.trades, 1);
        assert_eq!(bt.exit_tc, 1);
        assert!((bt.max_gain - expected).abs() < 1e-9, "max_gain: {}", bt.max_gain);
    }

    #[test]
    fn test_confidence_exit() {
        let n = 30;
        let (df, mut side) = long_fixture(n, 0.5);
        // confidence disappears from bar 6 on; the 5th zero bar is bar 10
        for i in 6..n {
            side.neg_conf[i] = 0.0;
        }

        let bt = backtest_performance(df, side, "test").unwrap();
        let expected = (110.0 / 101.0 - 1.0) * 100.0;
        assert_eq!(bt.trades, 1);
        assert_eq!(bt.exit_conf, 1);
        assert!((bt.max_gain - expected).abs() < 1e-9, "max_gain: {}", bt.max_gain);
    }

    #[test]
    fn test_tc_target_disables_confidence_exit() {
        // Same setup as test_confidence_exit (confidence gone from bar 6)
        // but the signal carries a tc target, so the trade must run to the
        // tc exit at bar 34 instead of closing at bar 10.
        let n = 60;
        let (df, mut side) = long_fixture(n, 0.5);
        for i in 6..n {
            side.neg_conf[i] = 0.0;
        }
        side.exit_time[1] = 1003.0;

        let bt = backtest_performance(df, side, "test").unwrap();
        let expected = (134.0 / 101.0 - 1.0) * 100.0;
        assert_eq!(bt.trades, 1);
        assert_eq!(bt.exit_conf, 0);
        assert_eq!(bt.exit_tc, 1);
        assert!((bt.max_gain - expected).abs() < 1e-9, "max_gain: {}", bt.max_gain);
    }

    #[test]
    fn test_stop_loss_exit() {
        let n = 20;
        let dates: Vec<String> = (0..n).map(|i| format!("d{}", i)).collect();
        let mut opens = vec![100.0; n];
        // entry at open[1] = 100; -21% at bar 5 breaches the 20% stop
        opens[2] = 95.0;
        opens[3] = 90.0;
        opens[4] = 85.0;
        opens[5] = 79.0;
        let time: Vec<f64> = (0..n).map(|i| 1000.0 + i as f64).collect();
        let df = polars::df!(
            "Date" => dates,
            "Ticker" => vec!["TST"; n],
            "Universe" => vec!["U"; n],
            "Open" => opens.clone(),
            "Close" => opens,
            "time" => time,
        )
        .unwrap();
        let mut buy = vec![0; n];
        buy[1] = 1;
        let side = BuySell {
            buy,
            sell: vec![0; n],
            pos_conf: vec![0.0; n],
            neg_conf: vec![0.5; n],
            exit_time: vec![f64::NAN; n],
        };

        let bt = backtest_performance(df, side, "test").unwrap();
        assert_eq!(bt.trades, 1);
        assert_eq!(bt.exit_stop, 1);
        assert!((bt.max_loss + 21.0).abs() < 1e-9, "max_loss: {}", bt.max_loss);
    }

    #[test]
    fn test_signal_fun_carries_tc() {
        // pos_conf above the level at bar 2 -> sell at bar 3 carrying pos_tc
        let df = polars::df!(
            "pos_conf" => &[0.0, 0.0, 0.9, 0.0, 0.0, 0.0],
            "neg_conf" => &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "pos_tc" => &[None, None, Some(1234.0), None, None, None],
            "neg_tc" => &[None::<f64>; 6],
        )
        .unwrap();
        let s = signal_fun(df, 0.5, 0.5);
        assert_eq!(s.sell, vec![0, 0, 0, -1, 0, 0]);
        assert_eq!(s.exit_time[3], 1234.0);
        assert!(s.exit_time[2].is_nan());
    }
}

pub async fn parquet_save_backtest(
    path: String,
    bt: Vec<Backtest>,
    univ: &str,
    ticker: String,
    production: bool,
) -> Result<(), Box<dyn StdError>> {
    // 2. Jsonify your struct Vec
    let json = serde_json::to_string(&bt)?;
    // 3. Create cursor from json
    let cursor = Cursor::new(json);
    // 4. Create polars DataFrame from reading cursor as json
    let mut df = JsonReader::new(cursor).finish()?;

    let folder = if production {
        "production".to_string()
    } else {
        "testing".to_string()
    };
    let file_path = match univ {
        "Crypto" => format!("{}/output_crypto/{}/{}.parquet", &path, folder, &ticker),
        _ => format!("{}/output/{}/{}.parquet", &path, folder, &ticker),
    };

    let mut file = File::create(file_path)?;
    ParquetWriter::new(&mut file).finish(&mut df)?;
    Ok(())
}
