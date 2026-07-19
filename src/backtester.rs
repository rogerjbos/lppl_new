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

    let pos = df.column("pos_conf").unwrap().f64().unwrap();
    let neg = df.column("neg_conf").unwrap().f64().unwrap();
    let _close = df.column("Close").unwrap().f64().unwrap();

    // println!("df: {:?}", df.clone());

    for i in 1..len - 1 {
        let pos_0: f64;
        let neg_0: f64;
        if i == len - 2 {
            pos_0 = pos.get(i + 1).unwrap();
            neg_0 = neg.get(i + 1).unwrap();
            // println!("pos {}  neg {}", pos_0, neg_0);
        } else {
            pos_0 = pos.get(i).unwrap();
            neg_0 = neg.get(i).unwrap();
        }
        // println!("i of {} : {} pos {} neg {}", i, len.clone(), pos_0, neg_0);

        if pos_0 > pos_level {
            sell[i + 1] = -1;
            // println!("sell triggered {} > {}", neg_0, neg_level);
        } else if neg_0 > neg_level {
            buy[i + 1] = 1;
            // println!("buy triggered {} > {}", neg_0, neg_level);
        }
        pos_conf[i + 1] = pos_0 as f32;
        neg_conf[i + 1] = neg_0 as f32;
    }
    BuySell {
        buy,
        sell,
        pos_conf,
        neg_conf,
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

pub fn summary_performance(df: DataFrame) -> Result<DataFrame, Box<dyn StdError>> {
    let out = df
        .lazy()
        .group_by_stable([col("strategy"), col("universe")])
        .agg([
            col("hit_ratio").mean().alias("hit_ratio"),
            col("realized_risk_reward").mean().alias("risk_reward"),
            col("avg_gain").mean().alias("avg_gain"),
            col("avg_loss").mean().alias("avg_loss"),
            col("max_gain").mean().alias("max_gain"),
            col("max_loss").mean().alias("max_loss"),
            col("buys").mean().alias("buys"),
            col("sells").mean().alias("sells"),
            col("trades").mean().alias("trades"),
            col("trades").sum().alias("total_trades"),
            col("profit_factor").count().alias("N"),
            col("expectancy").mean().alias("expectancy"),
            col("profit_factor").mean().alias("profit_factor"),
        ])
        // LPPLS signals are rare by design, so gate on total trades across
        // the universe (statistical validity), not mean trades per ticker.
        .filter(col("total_trades").gt_eq(lit(20)))
        .sort(
            vec!["hit_ratio"],
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
    // - An opposite signal closes the position at that bar's open and
    //   reverses into the new direction.
    // - A position is closed after MAX_HOLD_BARS bars, and any position
    //   still open at the end of the data is marked to market at the
    //   final close.
    // - The final bar's signal is derived from same-day confidence (kept
    //   for the production buy/sell list), so the backtest ignores it.
    let mut trade_results: Vec<f64> = Vec::new();
    let mut position: i32 = 0; // 0 = flat, 1 = long, -1 = short
    let mut entry_price = f64::NAN;
    let mut entry_bar: usize = 0;

    for i in 0..len {
        let is_last = i == len - 1;
        let buy_sig = !is_last && side.buy[i] == 1;
        let sell_sig = !is_last && side.sell[i] == -1;

        if position != 0 {
            let opposite = (position == 1 && sell_sig) || (position == -1 && buy_sig);
            let expired = i - entry_bar >= MAX_HOLD_BARS;
            if opposite || expired || is_last {
                let exit_price = if is_last {
                    close.get(i).unwrap_or(f64::NAN)
                } else {
                    open.get(i).unwrap_or(f64::NAN)
                };
                if entry_price > 0.0 && exit_price.is_finite() {
                    let ret = if position == 1 {
                        (exit_price / entry_price - 1.0) * 100.0
                    } else {
                        (entry_price - exit_price) / entry_price * 100.0
                    };
                    trade_results.push(ret);
                }
                position = 0;
                entry_price = f64::NAN;
            }
        }

        if position == 0 && (buy_sig || sell_sig) {
            let price = open.get(i).unwrap_or(f64::NAN);
            if price > 0.0 {
                position = if buy_sig { 1 } else { -1 };
                entry_price = price;
                entry_bar = i;
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
            f64::min(100., ag)
        }
    };
    let average_loss: f64 = {
        let al = sum_total_net_losses / total_net_losses.len() as f64;
        if al.is_nan() {
            0.0
        } else {
            f64::max(-100., al)
        }
    };
    let realized_risk_reward: f64 = {
        let rr = average_gain / average_loss;
        if rr.is_nan() {
            0.0
        } else {
            f64::min(100., rr)
        }
    };
    let trades: i32 = trade_results.len() as i32;

    // Expectancy
    let expectancy = {
        let ex = (average_gain * hit_ratio) - ((100. - hit_ratio) * average_loss);
        if ex.is_nan() {
            0.0
        } else {
            f64::min(999., ex)
        }
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
        };

        let bt = backtest_performance(df, side, "test").unwrap();

        // long: 100 -> 120 = +20%, short: 120 -> 90 (last close) = +25%
        assert_eq!(bt.trades, 2);
        assert!((bt.avg_gain - 22.5).abs() < 1e-9, "avg_gain: {}", bt.avg_gain);
        assert!((bt.max_gain - 25.0).abs() < 1e-9, "max_gain: {}", bt.max_gain);
        assert!((bt.hit_ratio - 100.0).abs() < 1e-9);
        assert_eq!(bt.buys, 1);
        assert_eq!(bt.sells, 2);
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
        };

        let bt = backtest_performance(df, side, "test").unwrap();

        // one long: 100 -> 130 (last close) = +30%
        assert_eq!(bt.trades, 1);
        assert!((bt.max_gain - 30.0).abs() < 1e-9, "max_gain: {}", bt.max_gain);
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

        let side = BuySell {
            buy,
            sell: vec![0; n],
            pos_conf: vec![0.0; n],
            neg_conf: vec![0.0; n],
        };

        let bt = backtest_performance(df, side, "test").unwrap();

        // Entered at open[1] = 101, expired MAX_HOLD_BARS later at
        // open[1 + MAX_HOLD_BARS] = 101 + 126 = 227.
        let expected = (227.0 / 101.0 - 1.0) * 100.0;
        assert_eq!(bt.trades, 1);
        assert!(
            (bt.max_gain - expected).abs() < 1e-9,
            "max_gain: {} expected: {}",
            bt.max_gain,
            expected
        );
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
