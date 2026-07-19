#![allow(non_snake_case)]
use chrono::{Duration, NaiveDate};
use polars::prelude::*;
use serde::Serialize;
use std::io::{Cursor, Error, ErrorKind};
use std::{
    collections::{BTreeMap, HashSet},
    env,
    error::Error as StdError,
    f64::consts::PI,
    fs::File,
    panic,
    path::Path,
    sync::Arc,
};
use tokio::fs;

pub mod backtester;
use crate::backtester::*;

pub mod clickhouse_mod;
use crate::clickhouse_mod::insert_score_dataframe;

pub mod argmin_mod;
use crate::argmin_mod::*;

#[derive(Debug, Serialize)]
pub struct FitResult {
    tc_d: NaiveDate,
    tc: f64,
    m: f64,
    w: f64,
    a: f64,
    b: f64,
    c: f64,
    c1: f64,
    c2: f64,
    t1_d: NaiveDate,
    t2_d: NaiveDate,
    t1: f64,
    t2: f64,
    O: f64,
    D: f64,
    p2: f64,
    best_cost: f64,
}

fn file_exists(path: &str) -> bool {
    Path::new(path).exists()
}

pub fn test_test() -> Result<(), Box<dyn StdError>> {
    let _ = test();
    Ok(())
}

pub async fn delete_all_files_in_folder<P: AsRef<Path>>(path: P) -> Result<(), Error> {
    let path = path.as_ref();

    if path.is_file() {
        // Attempt to delete the file and ignore "not found" errors
        if let Err(e) = fs::remove_file(path).await {
            if e.kind() != ErrorKind::NotFound {
                return Err(e);
            }
        }
    } else if path.is_dir() {
        // Attempt to read and delete contents in the directory and ignore "not found" errors
        let mut dir = match fs::read_dir(path).await {
            Ok(dir) => dir,
            Err(e) if e.kind() == ErrorKind::NotFound => return Ok(()), // Ignore if directory not found
            Err(e) => return Err(e),
        };

        while let Some(entry) = dir.next_entry().await? {
            let entry_path = entry.path();
            if entry_path.is_file() {
                if let Err(e) = fs::remove_file(entry_path).await {
                    if e.kind() != ErrorKind::NotFound {
                        return Err(e);
                    }
                }
            } else if entry_path.is_dir() {
                if let Err(e) = fs::remove_dir_all(entry_path).await {
                    if e.kind() != ErrorKind::NotFound {
                        return Err(e);
                    }
                }
            }
        }
    }
    Ok(())
}

pub fn min_max_scale_vec(observations: &Vec<f64>) -> Vec<f64> {
    let min = observations.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = observations
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    // Apply min-max scaling: (x - min) / (max - min)
    observations
        .iter()
        .map(|&x| (x - min) / (max - min))
        .collect()
}

// pub fn min_max_scale(observations: &[f64]) -> Vec<f64> {
//     let min = observations.iter().cloned().fold(f64::INFINITY, f64::min);
//     let max = observations.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
//     // Apply min-max scaling: (x - min) / (max - min)
//     observations.iter().map(|&x| (x - min) / (max - min)).collect()
// }

pub fn get_c(c1: f64, c2: f64) -> f64 {
    c1 / ((c2 / c1).atan()).cos()
}

pub fn get_oscillations(w: f64, tc: f64, t1: f64, t2: f64) -> f64 {
    (w / (2.0 * PI)) * ((tc - t1) / (tc - t2)).ln()
}

pub fn get_damping(m: f64, w: f64, b: f64, c: f64) -> f64 {
    (m * (b).abs()) / (w * (c).abs())
}

pub fn compute_nested_fits(
    time: &[f64],
    price: &[f64],
    window_size: usize,
    smallest_window_size: usize,
    outer_increment: usize,
    inner_increment: usize,
) -> Result<DataFrame, Box<dyn StdError>> {
    let mut res: Vec<FitResult> = Vec::new();
    let window_delta = window_size - smallest_window_size;

    if time.len() < window_size {
        return Err(Box::<dyn std::error::Error>::from("Too few observations."));
    }
    let obs_copy_len = time.len() - window_size;

    for i in (0..obs_copy_len + 1).step_by(outer_increment) {
        let time_slice = &time[i..i + window_size];
        let price_slice = &price[i..i + window_size];
        let p2 = *price_slice.last().unwrap();

        for j in (0..window_delta).step_by(inner_increment) {
            let time_shrinking_slice = &time_slice[j..window_size];
            let price_shrinking_slice = &price_slice[j..window_size];

            let nested_t1 = *time_shrinking_slice.first().unwrap();
            let nested_t2 = *time_shrinking_slice.last().unwrap();

            // Use a match or Result to handle potential errors
            if let Ok(result) =
                panic::catch_unwind(|| fit_argmin(time_shrinking_slice, price_shrinking_slice))
            {
                if let Ok((tc, m, w, a, b, c1, c2, best_cost)) = result {
                    let c = get_c(c1, c2);
                    let O = get_oscillations(w, tc, nested_t1, nested_t2);
                    let D = get_damping(m, w, b, c);

                    res.push(FitResult {
                        tc_d: ordinal_to_date(tc),
                        tc,
                        m,
                        w,
                        a,
                        b,
                        c,
                        c1,
                        c2,
                        t1_d: ordinal_to_date(nested_t1),
                        t2_d: ordinal_to_date(nested_t2),
                        t1: nested_t1,
                        t2: nested_t2,
                        O,
                        D,
                        p2,
                        best_cost,
                    });
                    // println!("i: {} j: {} best_cost: {}", i, j, best_cost);
                } else {
                    // println!("Error occurred at i: {} j: {}, skipping...", i, j);
                    print!("E");
                    continue;
                }
            } else {
                println!("Panic occurred at i: {} j: {}, skipping...", i, j);
                continue;
            }
        }
    }
    if res.is_empty() {
        return Err(Box::<dyn std::error::Error>::from("Nested fits failed"));
    }
    Ok(struct_to_df(res)?)
}

pub fn struct_to_df(res: Vec<FitResult>) -> Result<DataFrame, Box<dyn StdError>> {
    // 2. Jsonify your struct Vec
    let json = serde_json::to_string(&res).expect("results struct");
    // 3. Create cursor from json
    let cursor = Cursor::new(json);
    // 4. Create polars DataFrame from reading cursor as json
    let df = JsonReader::new(cursor).finish().expect("cursor");
    Ok(df)
}

pub fn compute_indicators(df: DataFrame) -> Result<DataFrame, Box<dyn StdError>> {
    // m tightened to (0.01, 0.99) so fits pinned at the theoretical
    // boundaries do not qualify as bubble evidence.
    let m_min = 0.01;
    let m_max = 0.99;
    let w_min = 2.0;
    let w_max = 15.0;
    let O_min = 2.5;
    let D_min = 0.5;
    // Relative-error gate: drop each ticker's worst fits by per-day MSE.
    // Within-ticker quantile rather than an absolute RMSE threshold because
    // absolute values are not comparable across assets: prices are min-max
    // scaled over the full history, so the same fit quality yields ~3x the
    // relative RMSE on a 1-year stock series vs a 5-year crypto series.
    let fit_quantile = 0.90;
    // Minimum number of qualified windows before a date's confidence is
    // nonzero — one lucky window out of two should not read as 50%.
    let min_qual = 2;

    let grouped = df
        .lazy()
        .with_columns(&[
            // tc_in_range column
            when(
                (col("tc").gt_eq(
                    when(
                        (col("t2") - lit(60.0))
                            .gt_eq(col("t2") - lit(0.5) * (col("t2") - col("t1"))),
                    )
                    .then(col("t2") - lit(60.0))
                    .otherwise(col("t2") - lit(0.5) * (col("t2") - col("t1"))),
                ))
                .and(
                    col("tc").lt_eq(
                        when(
                            (col("t2") + lit(252.0))
                                .lt_eq(col("t2") + lit(0.5) * (col("t2") - col("t1"))),
                        )
                        .then(col("t2") + lit(252.0))
                        .otherwise(col("t2") + lit(0.5) * (col("t2") - col("t1"))),
                    ),
                ),
            )
            .then(lit(true))
            .otherwise(lit(false))
            .alias("tc_in_range"),
            // m_in_range column
            (col("m").gt_eq(lit(m_min)).and(col("m").lt_eq(lit(m_max)))).alias("m_in_range"),
            // w_in_range column
            (col("w").gt_eq(lit(w_min)).and(col("w").lt_eq(lit(w_max)))).alias("w_in_range"),
            // O column
            when(col("b").neq(lit(0.0)).and(col("c").neq(lit(0.0))))
                .then(col("O"))
                .otherwise(lit(f64::INFINITY))
                .alias("O"),
            // D_in_range column
            (col("D").gt_eq(lit(D_min))).alias("D_in_range"),
            // O_in_range column
            (col("O").gt_eq(lit(O_min))).alias("O_in_range"),
            // fit_ok column: per-day MSE within the ticker's best fit_quantile
            ((col("best_cost") / (col("t2") - col("t1") + lit(1.0))).lt_eq(
                (col("best_cost") / (col("t2") - col("t1") + lit(1.0)))
                    .quantile(lit(fit_quantile), QuantileInterpolOptions::Linear),
            ))
            .alias("fit_ok"),
        ])
        .with_columns(&[
            // is_qualified column
            (col("tc_in_range")
                .and(col("m_in_range"))
                .and(col("w_in_range"))
                .and(col("O_in_range"))
                .and(col("D_in_range"))
                .and(col("fit_ok")))
            .alias("is_qualified"),
        ])
        .with_columns(&[
            col("b").lt(lit(0.0)).alias("pos_count"),
            // pos_qual_count column
            col("b")
                .lt(lit(0.0))
                .and(col("is_qualified"))
                .alias("pos_qual_count"),
            // neg_count column
            col("b").gt(lit(0.0)).alias("neg_count"),
            // neg_qual_count column
            col("b")
                .gt(lit(0.0))
                .and(col("is_qualified"))
                .alias("neg_qual_count"),
        ])
        .group_by([col("t2_d")])
        .agg([
            col("pos_count").sum().alias("pos_count_sum"),
            col("pos_qual_count").sum().alias("pos_qual_count_sum"),
            col("neg_count").sum().alias("neg_count_sum"),
            col("neg_qual_count").sum().alias("neg_qual_count_sum"),
            col("p2").max().alias("price"),
        ])
        .with_columns(&[
            when(
                col("pos_count_sum")
                    .gt(lit(0))
                    .and(col("pos_qual_count_sum").gt_eq(lit(min_qual))),
            )
            .then(
                col("pos_qual_count_sum").cast(DataType::Float64)
                    / col("pos_count_sum").cast(DataType::Float64),
            )
            .otherwise(lit(0.0))
            .alias("pos_conf"),
            when(
                col("neg_count_sum")
                    .gt(lit(0))
                    .and(col("neg_qual_count_sum").gt_eq(lit(min_qual))),
            )
            .then(
                col("neg_qual_count_sum").cast(DataType::Float64)
                    / col("neg_count_sum").cast(DataType::Float64),
            )
            .otherwise(lit(0.0))
            .alias("neg_conf"),
        ])
        .collect()?;

    Ok(grouped)
}

/// Indicator days to skip before emitting eps_norm: the running-max
/// normalization makes the first observation +/-1 by construction, so the
/// early values are meaningless until the max has stabilized.
pub const RESIDUAL_BURN_IN: usize = 20;

/// Normalized LPPL residual indicator (arXiv:2510.10878). For each window
/// end date t2, take the median residual of the observed (scaled log)
/// price vs the fitted LPPLS value across the nested windows, then
/// normalize by the running max of |residual| so eps_norm is in [-1, 1].
/// The paper models residuals as an Ornstein-Uhlenbeck process to justify
/// boundedness/stationarity; the tradable indicator is eps_norm with a
/// threshold (paper: 0.8) and a minimum duration (paper: 10 days).
/// Positive eps_norm = price above the fitted bubble trajectory.
pub fn compute_residual_indicator(df: &DataFrame) -> Result<DataFrame, Box<dyn StdError>> {
    let t2d = df.column("t2_d")?.date()?;
    let t2 = df.column("t2")?.f64()?;
    let p2 = df.column("p2")?.f64()?;
    let tc = df.column("tc")?.f64()?;
    let m = df.column("m")?.f64()?;
    let w = df.column("w")?.f64()?;
    let a = df.column("a")?.f64()?;
    let b = df.column("b")?.f64()?;
    let c1 = df.column("c1")?.f64()?;
    let c2 = df.column("c2")?.f64()?;

    let mut by_date: BTreeMap<i32, Vec<f64>> = BTreeMap::new();
    for i in 0..df.height() {
        let (
            Some(d),
            Some(t2v),
            Some(p2v),
            Some(tcv),
            Some(mv),
            Some(wv),
            Some(av),
            Some(bv),
            Some(c1v),
            Some(c2v),
        ) = (
            t2d.get(i),
            t2.get(i),
            p2.get(i),
            tc.get(i),
            m.get(i),
            w.get(i),
            a.get(i),
            b.get(i),
            c1.get(i),
            c2.get(i),
        )
        else {
            continue;
        };
        // Same |tc - t| clamp as the fitter, so points near tc cannot NaN
        let dt = (tcv - t2v).abs().max(1e-8);
        let log_dt = dt.ln();
        let fitted =
            av + dt.powf(mv) * (bv + c1v * (wv * log_dt).cos() + c2v * (wv * log_dt).sin());
        let eps = p2v - fitted;
        if eps.is_finite() {
            by_date.entry(d).or_default().push(eps);
        }
    }

    let mut dates: Vec<i32> = Vec::with_capacity(by_date.len());
    let mut eps_norm: Vec<f64> = Vec::with_capacity(by_date.len());
    let mut run_max = 0.0f64;
    for (idx, (d, mut v)) in by_date.into_iter().enumerate() {
        v.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let n = v.len();
        let med = if n % 2 == 1 {
            v[n / 2]
        } else {
            0.5 * (v[n / 2 - 1] + v[n / 2])
        };
        run_max = run_max.max(med.abs());
        let e = if idx < RESIDUAL_BURN_IN || run_max <= 0.0 {
            0.0
        } else {
            med / run_max
        };
        dates.push(d);
        eps_norm.push(e);
    }

    let out = df!("t2_d_i" => dates, "eps_norm" => eps_norm)?
        .lazy()
        .with_column(col("t2_d_i").cast(DataType::Date).alias("t2_d"))
        .select([col("t2_d"), col("eps_norm")])
        .collect()?;
    Ok(out)
}

pub fn lppls(t: f64, tc: f64, m: f64, w: f64, a: f64, b: f64, c1: f64, c2: f64) -> f64 {
    a + (tc - t).powf(m) * (b + c1 * (w * (tc - t).ln()).cos() + c2 * (w * (tc - t).ln()).sin())
}

pub fn compute_sse(
    time: &Vec<f64>,
    price: &Vec<f64>,
    tc: f64,
    m: f64,
    w: f64,
    a: f64,
    b: f64,
    c1: f64,
    c2: f64,
) -> f64 {
    let lppls_values: Vec<f64> = time
        .iter()
        .map(|&t| lppls(t, tc, m, w, a, b, c1, c2))
        .collect();
    // Perform element-wise subtraction and compute the sum of squared errors
    lppls_values
        .iter()
        .zip(price.iter()) // Iterate over both time and price simultaneously
        .map(|(lppls_val, &price_val)| (lppls_val - price_val).powi(2)) // Square the differences
        .sum()
}

// Function to convert a NaiveDate to days since CE
pub fn date_to_ce(date: NaiveDate) -> f64 {
    let zero_date = NaiveDate::from_ymd_opt(1, 1, 1).expect("valid CE zero date");
    (date - zero_date).num_days() as f64
}
// Function to convert days since CE back to NaiveDate
pub fn ce_to_date(days: f64) -> NaiveDate {
    NaiveDate::from_ymd_opt(1970, 1, 1).expect("valid Unix epoch") + Duration::days(days as i64)
}
// Function to convert days since 1970-01-01 back to NaiveDate
pub fn ordinal_to_date(days: f64) -> NaiveDate {
    NaiveDate::from_ymd_opt(1, 1, 1).expect("valid Unix epoch") + Duration::days(days as i64)
}

pub fn load_example_data() -> Result<LazyFrame, Box<dyn StdError>> {
    let mut schema = Schema::with_capacity(7);
    schema.with_column("Date".into(), DataType::Date);
    schema.with_column("Open".into(), DataType::Float64);
    schema.with_column("High".into(), DataType::Float64);
    schema.with_column("Low".into(), DataType::Float64);
    schema.with_column("Close".into(), DataType::Float64);
    schema.with_column("Adj Close".into(), DataType::Float64);
    schema.with_column("Volume".into(), DataType::Float64);

    let lf = LazyCsvReader::new("/Users/rogerbos/rust_home/lppl_new/nasdaq_dotcom.csv")
        .with_has_header(true)
        .with_schema(Some(Arc::new(schema)))
        // .with_truncate_ragged_lines(true) // Allow extra columns in CSV
        .finish()?
        .with_column(
            col("Date")
                .map(
                    |s| {
                        let chunked = s
                            .date() // Directly access date data
                            .expect("series must contain dates")
                            .into_iter()
                            .map(|opt_date| {
                                opt_date.map(|days| date_to_ce(ce_to_date(days.into())))
                            })
                            .collect::<Float64Chunked>();
                        Ok(Some(Series::new("time".into(), chunked)))
                    },
                    GetOutput::from_type(DataType::Float64),
                )
                .alias("time"),
        )
        .with_column(
            col("Adj Close")
                .map(
                    |s| {
                        let chunked = s
                            .f64() // Access the "Adj Close" column as f64
                            .expect("series must contain f64 data")
                            .into_iter()
                            .map(|opt_price| {
                                opt_price.map(|price| price.ln()) // Take natural log of each price
                            })
                            .collect::<Float64Chunked>();
                        Ok(Some(Series::new("price".into(), chunked))) // Create the new "price" column
                    },
                    GetOutput::from_type(DataType::Float64),
                )
                .alias("price"),
        );
    Ok(lf)
}

pub async fn run_fits(lf: LazyFrame, tag: &str) -> Result<(), Box<dyn StdError>> {
    let user_path = match env::var("CLICKHOUSE_USER_PATH") {
        Ok(path) => path,
        Err(_) => String::from("/srv"),
    };

    // For testing use all the data, for production only use the last 120 days
    let df = if tag == "testing" {
        lf.collect()?
    } else {
        lf.collect()?.tail(Some(120))
    };
    // println!("df: {}", df.clone());

    let ticker1 = df.column("Ticker").unwrap().get(0).unwrap().to_string();
    let ticker = ticker1.trim_matches('"').to_string();
    let universe1 = df.column("Universe").unwrap().get(0).unwrap().to_string();
    let universe = universe1.trim_matches('"').to_string();

    let fname = format!(
        "{}/rust_home/lppl_new/fit/{}/fit_{}_{}.csv",
        user_path.to_string(),
        tag,
        universe,
        ticker
    );

    let time_series = df.column("time")?;
    let time: Vec<f64> = time_series.f64()?.into_no_null_iter().collect();

    let price_series = df.column("scaled_price_ln")?;
    let price: Vec<f64> = price_series.f64()?.into_no_null_iter().collect();

    let res = compute_nested_fits(&time, &price, 120, 30, 1, 5)?;
    // println!("res: {:?}", res.select(["t2_d", "tc"])?);

    // println!("fname: {}", &fname);
    let mut file = File::create(&fname)?;
    CsvWriter::new(&mut file).finish(&mut res.clone())?; // Use clone to avoid moving `res`
    Ok(())
}

pub async fn run_backtests(
    lf: LazyFrame,
    tag: &str,
    production: bool,
) -> Result<Vec<Backtest>, Box<dyn StdError>> {
    let user_path = match env::var("CLICKHOUSE_USER_PATH") {
        Ok(path) => path,
        Err(_) => String::from("/srv"),
    };

    let production_str = if production { "production" } else { "testing" };
    let df = lf.collect()?;
    // println!("df: {:?}", df.clone());

    let ticker1 = df.column("Ticker").unwrap().get(0).unwrap().to_string();
    let ticker = ticker1.trim_matches('"').to_string();
    let universe1 = df.column("Universe").unwrap().get(0).unwrap().to_string();
    let universe = universe1.trim_matches('"').to_string();

    let fname = format!(
        "{}/rust_home/lppl_new/fit/{}/fit_{}_{}.csv",
        user_path.to_string(),
        production_str,
        universe,
        ticker
    );

    let mut schema = Schema::with_capacity(17);
    schema.with_column("tc_d".into(), DataType::Date);
    schema.with_column("tc".into(), DataType::Float64);
    schema.with_column("m".into(), DataType::Float64);
    schema.with_column("w".into(), DataType::Float64);
    schema.with_column("a".into(), DataType::Float64);
    schema.with_column("b".into(), DataType::Float64);
    schema.with_column("c".into(), DataType::Float64);
    schema.with_column("c1".into(), DataType::Float64);
    schema.with_column("c2".into(), DataType::Float64);
    schema.with_column("t1_d".into(), DataType::Date);
    schema.with_column("t2_d".into(), DataType::Date);
    schema.with_column("t1".into(), DataType::Float64);
    schema.with_column("t2".into(), DataType::Float64);
    schema.with_column("O".into(), DataType::Float64);
    schema.with_column("D".into(), DataType::Float64);
    schema.with_column("p2".into(), DataType::Float64);
    schema.with_column("best_cost".into(), DataType::Float64);

    if file_exists(&fname) {
        // println!("File exists!");

        let res = LazyCsvReader::new(&fname)
            .with_has_header(true)
            .with_schema(Some(Arc::new(schema)))
            .finish()?
            .collect();
        // println!("res: {:?}", &res);

        let fits_df = res?;
        let res_ind = compute_residual_indicator(&fits_df)?;
        let ind = compute_indicators(fits_df)?.select(["t2_d", "pos_conf", "neg_conf"])?;
        let out = df
            .left_join(&ind, ["Date"], ["t2_d"])?
            .left_join(&res_ind, ["Date"], ["t2_d"])?
            .lazy()
            .with_column(cols(["pos_conf", "neg_conf", "eps_norm"]).fill_null(lit(0.)));
        // println!("out with ind: {:?}", &out.clone().collect());

        // needs to be awaited
        let mut signals: Vec<Signal> = Vec::new();
        // println!("tag: {}", &tag);

        if !production {
            // testing

            let values: Vec<f64> = std::iter::once(0.01)
                .chain((5..=80).step_by(5).map(|x| x as f64 / 100.0))
                .collect();

            for i in &values {
                for j in &values {
                    let param1 = *i;
                    let param2 = *j;
                    let name = format!("lppl_{:.2}_{:.2}", param1, param2);

                    // Use a closure to capture param1 and param2
                    let function_with_params =
                        move |df: DataFrame| -> BuySell { signal_fun(df, param1, param2) };

                    signals.push(Signal {
                        name,
                        param1,
                        param2,
                        f: Arc::new(function_with_params), // Store the closure in the Arc
                    });
                }
            }

            // Normalized-residual strategies (arXiv:2510.10878): long when
            // eps_norm <= -tau sustained for dmin days, short when >= +tau.
            // Paper uses tau = 0.7-0.8 entries with a 10-day minimum.
            for tau in [0.5, 0.6, 0.7, 0.8, 0.9] {
                for dmin in [1usize, 5, 10] {
                    let name = format!("res_{:.1}_{}", tau, dmin);
                    let function_with_params =
                        move |df: DataFrame| -> BuySell { res_signal_fun(df, tau, dmin) };

                    signals.push(Signal {
                        name,
                        param1: tau,
                        param2: dmin as f64,
                        f: Arc::new(function_with_params),
                    });
                }
            }
        } else {
            // production

            // Miniimum confidence to be included in the buy / sell list
            let param1: f64 = 0.01;
            let param2: f64 = 0.01;
            let name = format!("lppl_{:.2}_{:.2}", param1, param2).to_string();

            // Use a closure to capture param1 and param2
            let function_with_params =
                move |df: DataFrame| -> BuySell { signal_fun(df, param1, param2) };

            signals.push(Signal {
                name: name,
                param1: param1,
                param2: param2,
                f: Arc::new(function_with_params),
            });

            // Recalibrated 2026-07-18 (percent-return backtest, position-based
            // exits, subordinated 3D fits). SC had NO profitable (p1, p2) cell
            // in the grid, so thresholds are set to the least-active corner
            // (fires almost never) rather than a losing strategy.
            if tag == "sc" || tag == "SC1" || tag == "SC2" || tag == "SC3" || tag == "SC4" {
                let param1: f64 = 0.80;
                let param2: f64 = 0.80;
                let name = format!("lppl_{:.2}_{:.2}", param1, param2).to_string();

                // Use a closure to capture param1 and param2
                let function_with_params =
                    move |df: DataFrame| -> BuySell { signal_fun(df, param1, param2) };

                signals.push(Signal {
                    name: name,
                    param1: param1,
                    param2: param2,
                    f: Arc::new(function_with_params),
                });
            } else if tag == "micro"
                || tag == "Micro1"
                || tag == "Micro2"
                || tag == "Micro3"
                || tag == "Micro4"
            {
                // Recalibrated 2026-07-18: Micro had NO profitable (p1, p2)
                // cell (worst group in the grid); least-active corner chosen.
                let param1: f64 = 0.80;
                let param2: f64 = 0.80;
                let name = format!("lppl_{:.2}_{:.2}", param1, param2).to_string();

                // Use a closure to capture param1 and param2
                let function_with_params =
                    move |df: DataFrame| -> BuySell { signal_fun(df, param1, param2) };

                signals.push(Signal {
                    name: name,
                    param1: param1,
                    param2: param2,
                    f: Arc::new(function_with_params),
                });
            } else if tag == "mc" || tag == "MC1" || tag == "MC2" {
                // Recalibrated 2026-07-19 (quantile fit gate, m 0.01-0.99,
                // min 2 qualified windows): positive band at p2 0.15,
                // p1 0.60-0.80 (expectancy up to +18, ~44-50 trades).
                let param1: f64 = 0.75;
                let param2: f64 = 0.15;
                let name = format!("lppl_{:.2}_{:.2}", param1, param2).to_string();

                // Use a closure to capture param1 and param2
                let function_with_params =
                    move |df: DataFrame| -> BuySell { signal_fun(df, param1, param2) };

                signals.push(Signal {
                    name: name,
                    param1: param1,
                    param2: param2,
                    f: Arc::new(function_with_params),
                });
            } else if tag == "lc" || tag == "LC1" || tag == "LC2" {
                // Recalibrated 2026-07-19 (quantile fit gate, m 0.01-0.99,
                // min 2 qualified windows): stable positive region at
                // p1 0.65-0.75, p2 0.15-0.30 (expectancy +18 to +28);
                // 0.70/0.20 is the consensus center across three passes.
                let param1: f64 = 0.70;
                let param2: f64 = 0.20;
                let name = format!("lppl_{:.2}_{:.2}", param1, param2).to_string();

                // Use a closure to capture param1 and param2
                let function_with_params =
                    move |df: DataFrame| -> BuySell { signal_fun(df, param1, param2) };

                signals.push(Signal {
                    name: name,
                    param1: param1,
                    param2: param2,
                    f: Arc::new(function_with_params),
                });
            } else if tag == "crypto" || tag == "Crypto" {
                // Recalibrated 2026-07-19 (2020+ history): NO profitable
                // (p1, p2) cell out of 289 — contrarian entries with fixed
                // exits lose in both directions on crypto (shorting pumps,
                // buying crashes). Least-active corner chosen; the old
                // 0.55/0.65 scored -1152 expectancy over 666 trades.
                let param1: f64 = 0.80;
                let param2: f64 = 0.80;
                let name = format!("lppl_{:.2}_{:.2}", param1, param2).to_string();

                // Use a closure to capture param1 and param2
                let function_with_params =
                    move |df: DataFrame| -> BuySell { signal_fun(df, param1, param2) };

                signals.push(Signal {
                    name: name,
                    param1: param1,
                    param2: param2,
                    f: Arc::new(function_with_params),
                });
            }
        }
        // println!("tag: {} signals: {:?}", tag, signals);
        Ok(run_all_backtests(out, signals).await?)
    } else {
        return Err(Box::<dyn std::error::Error>::from("Fit file missing"));
    }
}

pub async fn backtest_helper(
    path: String,
    u: &str,
    batch_size: usize,
    production: bool,
) -> Result<(), Box<dyn StdError>> {
    println!("Backtest starting: {}", &u);
    let folder = if production { "production" } else { "testing" };
    let file_path = format!("{}/data/{}/{}.csv", path, folder, u);
    // println!("backtest_helper file_path: {}", file_path);

    let lf = read_price_file(file_path).await?;

    // println!("backtest_helper lf: {:?}", lf.clone().collect());
    // Collect the unique tickers into a DataFrame
    let unique_tickers_df = lf
        .clone()
        .select([col("Ticker").unique().alias("unique_tickers")])
        .collect()?;

    // Assuming the 'unique_tickers' column is of type Utf8
    let unique_tickers_series = unique_tickers_df.column("unique_tickers")?;

    let output = if u == "Crypto" {
        "output_crypto"
    } else {
        "output"
    };
    let dir_path = format!("{}/{}/{}", path, output, folder);
    // println!("backtest_helper dir_path: {}", dir_path);

    let mut filenames: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(dir_path)? {
        let entry = entry?;
        let path = entry.path();
        // println!("backtest_helper path: {:?}", path);

        // Check if the entry is a file and has a `.parquet` extension
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("parquet") {
            // Convert the file stem to a String and push it into the filenames vector
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                filenames.push(stem.to_owned());
            }
        }
    }

    // Convert filenames to a HashSet
    let filenames_set: HashSet<String> = filenames.into_iter().collect();

    // Filter out tickers that are already done
    let needed: Vec<String> = unique_tickers_series
        .str()?
        .into_iter()
        .filter_map(|value| value.map(|v| v.to_string()))
        .filter(|ticker| !filenames_set.contains(ticker))
        // .take(5) // used for testing purposes
        .collect();
    // let needed = ["00".to_string(), "ada".to_string(), "dot".to_string()];
    // let needed = ["MAC".to_string(), "CNK".to_string(), "RDN".to_string()];
    // println!("needed: {:?}", &needed);

    let out_of = needed.len();
    let mut remaining = out_of;

    for i in (0..needed.len()).step_by(batch_size) {
        let last = if remaining < batch_size {
            remaining
        } else {
            batch_size
        };
        let unique_tickers = &needed[i..i + last];
        // collect futures for processing each ticker
        let futures: Vec<_> = unique_tickers
            .into_iter()
            .map(|ticker| {
                // clone outside the async block
                let lf_clone = lf.clone();
                let ticker_clone: String = ticker.clone();
                let path_clone: String = path.clone();

                async move {
                    let filtered_lf = lf_clone.filter(col("Ticker").eq(lit(ticker.to_string())));
                    let tag: &str = match (production, u) {
                        (false, _) => "testing",
                        (true, "Crypto") => "crypto",
                        (true, u) if u.starts_with("Micro") => "micro",
                        (true, u) if u.starts_with("SC") => "sc",
                        (true, u) if u.starts_with("MC") => "mc",
                        (true, u) if u.starts_with("LC") => "lc",
                        (_, _) => "crypto",
                    };
                    println!(
                        "LPPL running {} '{}' backtests: {} of {}",
                        u,
                        ticker_clone,
                        out_of - remaining,
                        out_of
                    );
                    // println!("lf: {:?}", filtered_lf.clone().collect());

                    match run_backtests(filtered_lf, tag, production).await {
                        Ok(backtest_results) => {
                            // println!("<==============================================================>bt: {:?}", &backtest_results);
                            if let Err(e) = parquet_save_backtest(
                                path_clone,
                                backtest_results,
                                u,
                                ticker_clone,
                                production,
                            )
                            .await
                            {
                                eprintln!("Error saving backtest to parquet: {}", e);
                            }
                        }
                        Err(e) => eprintln!("Skipping '{}' backtest: {}", ticker_clone, e),
                    }
                }
            })
            .collect();
        // await all futures to complete
        futures::future::join_all(futures).await;
        remaining -= last;
    }
    println!("Backtest done: {}", &u);

    Ok(())
}

pub async fn score(datetag: &str, stocks: bool) -> Result<(), Box<dyn StdError>> {
    // read in the testing file to get the historical performance for scoring
    let user_path = match env::var("CLICKHOUSE_USER_PATH") {
        Ok(path) => path,
        Err(_) => String::from("/srv"),
    };
    let path = format!("{}/rust_home/lppl_new/performance", user_path);
    let path: &str = &path;

    // let path: &str = "/Users/rogerbos/rust_home/backtester";
    let tag = if stocks { "stocks" } else { "crypto" };

    // println!("ath: {}", &path);
    // println!("tag: {}", &tag);
    let file_path = format!("{}/{}_testing.csv", path, tag);
    // println!("file_path: {}", &file_path);
    let testing = LazyCsvReader::new(file_path).finish()?;
    // println!("testing: {:?}", testing.clone().collect());

    // Manually create the schema and add fields
    let mut buysell_schema = Schema::with_capacity(8);
    buysell_schema.with_column("ticker".into(), DataType::String);
    buysell_schema.with_column("universe".into(), DataType::String);
    buysell_schema.with_column("strategy".into(), DataType::String);
    buysell_schema.with_column("date".into(), DataType::Date);
    buysell_schema.with_column("buy".into(), DataType::Int64);
    buysell_schema.with_column("sell".into(), DataType::Int64);
    buysell_schema.with_column("pos_conf".into(), DataType::Float64);
    buysell_schema.with_column("neg_conf".into(), DataType::Float64);
    let buysell_schema = Arc::new(buysell_schema);

    // read in the buys
    let buy_path = format!("{}/{}_buys_{}.csv", path, tag, datetag);
    let buys = LazyCsvReader::new(buy_path)
        .with_schema(Some(buysell_schema.clone()))
        .with_has_header(true)
        .finish()?
        .join(
            testing.clone(),
            vec![col("strategy"), col("universe")],
            vec![col("strategy"), col("universe")],
            JoinType::Left.into(),
        )
        .group_by_stable([col("date"), col("universe"), col("ticker")])
        .agg([
            col("buy").sum().alias("side"),
            col("risk_reward").sum().alias("risk_reward"),
            col("expectancy").sum().alias("expectancy"),
            col("profit_factor").sum().alias("profit_factor"),
            col("pos_conf").mean().alias("pos_conf"),
            col("neg_conf").mean().alias("neg_conf"),
        ])
        .sort(
            vec!["neg_conf"],
            SortMultipleOptions {
                descending: vec![false],
                nulls_last: vec![true],
                ..Default::default()
            },
        );

    // read in the sells
    let sell_path = format!("{}/{}_sells_{}.csv", path, tag, datetag);
    let sells = LazyCsvReader::new(sell_path)
        .with_schema(Some(buysell_schema))
        .with_has_header(true)
        .finish()?
        .join(
            testing,
            vec![col("strategy"), col("universe")],
            vec![col("strategy"), col("universe")],
            JoinType::Left.into(),
        )
        .group_by_stable([col("date"), col("universe"), col("ticker")])
        .agg([
            col("buy").sum().alias("side"),
            col("risk_reward").sum().alias("risk_reward"),
            col("expectancy").sum().alias("expectancy"),
            col("profit_factor").sum().alias("profit_factor"),
            col("pos_conf").mean().alias("pos_conf"),
            col("neg_conf").mean().alias("neg_conf"),
        ])
        .sort(
            vec!["pos_conf"],
            SortMultipleOptions {
                descending: vec![false],
                nulls_last: vec![true],
                ..Default::default()
            },
        );

    let both = concat(&[buys, sells], Default::default())?
        .sort(
            vec!["pos_conf", "neg_conf"],
            SortMultipleOptions {
                descending: vec![false, true],
                nulls_last: vec![true, true],
                ..Default::default()
            },
        )
        .collect()?;
    // println!("both: {:?}", both);
    // println!("both: {:?}", both.get_column_names());

    let path = format!("{}/rust_home/lppl_new", user_path);
    let both_path = format!("{}/score/{}_{}.csv", path, tag, datetag);
    // println!("both_path: {:?}", both_path);
    let mut file = File::create(both_path)?;
    let _ = CsvWriter::new(&mut file).finish(&mut both.clone());

    if both.height() > 0 {
        if let Err(e) = insert_score_dataframe(both).await {
            eprintln!("Error in insert_score_dataframe: {}", e);
        }
    } else {
        println!("No observations: skipping insert.");
    }

    Ok(())
}

pub async fn read_price_file(file_path: String) -> Result<LazyFrame, Box<dyn StdError>> {
    // Manually create the schema and add fields
    let mut schema = Schema::with_capacity(8);
    schema.with_column("Date".into(), DataType::Date);
    schema.with_column("Ticker".into(), DataType::String);
    schema.with_column("Universe".into(), DataType::String);
    schema.with_column("Open".into(), DataType::Float64);
    schema.with_column("High".into(), DataType::Float64);
    schema.with_column("Low".into(), DataType::Float64);
    schema.with_column("Close".into(), DataType::Float64);
    schema.with_column("Volume".into(), DataType::Float64);
    let schema = Arc::new(schema);

    // READ IN PRICE FILE
    let lf: LazyFrame = LazyCsvReader::new(file_path)
        .with_schema(Some(schema))
        .with_has_header(true)
        .finish()?
        // Drop rows where ln() would be undefined and that would otherwise
        // poison the min-max scaling below.
        .filter(col("Close").is_not_null().and(col("Close").gt(lit(0.0))))
        .with_column(
            col("Date")
                .map(
                    |s| {
                        let chunked = s
                            .date()
                            .expect("series must contain dates")
                            .into_iter()
                            .map(|opt_date| {
                                opt_date.map(|days| date_to_ce(ce_to_date(days.into())))
                            })
                            .collect::<Float64Chunked>();
                        Ok(Some(Series::new("time".into(), chunked)))
                    },
                    GetOutput::from_type(DataType::Float64),
                )
                .alias("time"),
        )
        .with_column(
            // LPPLS is defined on log price, so always take ln(). Negative
            // log prices (sub-$1 assets) are perfectly valid; rows with
            // null or non-positive Close are filtered out above.
            col("Close")
                .map(
                    |s| {
                        let chunked = s
                            .f64()
                            .expect("series must contain f64 data")
                            .into_iter()
                            .map(|opt_price| opt_price.map(|price| price.ln()))
                            .collect::<Float64Chunked>();
                        Ok(Some(Series::new("price_ln".into(), chunked)))
                    },
                    GetOutput::from_type(DataType::Float64),
                )
                .alias("price_ln"),
        )
        .with_column(
            col("Close")
                .map(
                    |s| {
                        let chunked = s.f64().expect("series must contain f64 data");
                        let values: Vec<f64> = chunked
                            .into_iter()
                            .map(|opt_price| opt_price.unwrap_or(0.0))
                            .collect();
                        let scaled = min_max_scale_vec(&values);
                        Ok(Some(Series::new("scaled_price".into(), scaled)))
                    },
                    GetOutput::from_type(DataType::Float64),
                )
                .alias("scaled_price"),
        )
        .with_column(
            col("price_ln")
                .map(
                    |s| {
                        let chunked = s.f64().expect("series must contain f64 data");
                        let values: Vec<f64> = chunked
                            .into_iter()
                            .map(|opt_price| opt_price.unwrap_or(0.0))
                            .collect();
                        let scaled = min_max_scale_vec(&values);
                        Ok(Some(Series::new("scaled_price_ln".into(), scaled)))
                    },
                    GetOutput::from_type(DataType::Float64),
                )
                .alias("scaled_price_ln"),
        );

    Ok(lf)
}

pub async fn fits_helper(
    path: String,
    u: &str,
    batch_size: usize,
    production: bool,
) -> Result<(), Box<dyn StdError>> {
    println!("Compute nested fits starting: {}", &u);
    let folder = if production { "production" } else { "testing" };
    let file_path = format!("{}/data/{}/{}.csv", path, folder, u);
    // println!("here 0 file_path: {:?}", &file_path);

    let lf = read_price_file(file_path).await?;
    println!("lf: {:?}", lf.clone().collect());

    // Collect the unique tickers into a DataFrame
    let unique_tickers_df = lf
        .clone()
        .select([col("Ticker").unique().alias("unique_tickers")])
        .collect()?;

    // Assuming the 'unique_tickers' column is of type Utf8
    let unique_tickers_series = unique_tickers_df.column("unique_tickers")?;

    let output = "fit"; // if u == "Crypto" { "fit" } else { "fit" };
    let dir_path = format!("{}/{}/{}", path, output, folder);

    let mut filenames: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(dir_path)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("csv") {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                // Split the stem by underscores and collect the parts
                let parts: Vec<&str> = stem.split('_').collect();

                // Check if there are at least 3 parts (before the 2nd underscore)
                if parts.len() >= 3 {
                    // The symbol is the third part after the second underscore
                    let symbol = parts[2];
                    filenames.push(symbol.to_owned());
                }
            }
        }
    }

    // Convert filenames to a HashSet
    let filenames_set: HashSet<String> = filenames.into_iter().collect();
    // println!("filenames_set: {:?}", &filenames_set);

    // Filter out tickers that are already done
    let needed: Vec<String> = unique_tickers_series
        .str()?
        .into_iter()
        .filter_map(|value| value.map(|v| v.to_string()))
        .filter(|ticker| !filenames_set.contains(ticker))
        // .take(5) // used for testing
        .collect();
    // let needed = ["00", "ada", "dot"];
    // println!("needed: {:?}", &needed);

    let out_of = needed.len();
    let mut remaining = out_of;

    for i in (0..needed.len()).step_by(batch_size) {
        let last = if remaining < batch_size {
            remaining
        } else {
            batch_size
        };
        let unique_tickers = &needed[i..i + last];

        // collect tasks to be run in parallel
        let futures: Vec<_> = unique_tickers
            .iter()
            .map(|ticker| {
                let lf_clone = lf.clone();
                let ticker_clone = ticker.clone();
                let u_clone = u.to_string(); // to pass it into the async block
                                             // println!("u u: {}", u);

                tokio::spawn(async move {
                    let filtered_lf =
                        lf_clone.filter(col("Ticker").eq(lit(ticker_clone.to_string())));
                    let tag: &str = match (production, u_clone.as_str()) {
                        (false, _) => "testing",
                        // (true, "Crypto") => "crypto",
                        // (true, "Micro") => "micro",
                        // (true, "SC") => "sc",
                        // (true, "MC") => "mc",
                        // (true, "LC") => "lc",
                        (_, _) => "production",
                    };
                    println!(
                        "Running {} '{}' compute nested fits: {} of {}",
                        u_clone,
                        ticker_clone,
                        out_of - remaining,
                        out_of
                    );
                    // println!("filtered_lf: {:?}", filtered_lf.clone().collect());
                    match run_fits(filtered_lf, tag).await {
                        Ok(_fit_results) => {
                            // handle results here
                        }
                        Err(e) => eprintln!(
                            "Error running '{}' compute nested fits: {}",
                            ticker_clone, e
                        ),
                    }
                })
            })
            .collect();

        // Await all tasks and handle errors
        futures::future::join_all(futures).await;

        remaining -= last;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test for get_c
    #[test]
    fn test_get_c() {
        // Test 1: Positive values
        let result = get_c(2.0, 1.0);
        let expected = 2.0 / ((1.0_f64 / 2.0).atan()).cos();
        assert!((result - expected).abs() < 1e-12);

        // Test 2: Negative values
        let result = get_c(-2.0, -1.0);
        let expected = -2.0 / ((0.5_f64).atan()).cos();
        assert!((result - expected).abs() < 1e-12);

        // Test 3: c2 == 0 leaves c1 unchanged
        let result = get_c(1.0, 0.0);
        assert!((result - 1.0).abs() < 1e-12);
    }

    // Test for get_oscillations
    #[test]
    fn test_get_oscillations() {
        // Test 1: Regular values
        let w = 6.28;
        let result = get_oscillations(w, 10.0, 2.0, 8.0);
        let expected = (w / (2.0 * PI)) * ((10.0_f64 - 2.0) / (10.0 - 8.0)).ln();
        assert!((result - expected).abs() < 1e-12);

        // Test 2: tc == t1 makes the ratio 0, and ln(0) is -infinity
        let result = get_oscillations(w, 5.0, 5.0, 3.0);
        assert!(result.is_infinite() && result < 0.0);
    }

    // Test for get_damping
    #[test]
    fn test_get_damping() {
        // Test 1: Regular values
        let result = get_damping(1.0, 2.0, 0.5, 1.0);
        assert!((result - 0.25).abs() < 1e-12);

        // Test 2: Negative values (b and c enter as absolute values)
        let result = get_damping(-1.0, 2.0, -0.5, -1.0);
        assert!((result + 0.25).abs() < 1e-12);

        // Test 3: Zero damping
        let result = get_damping(0.0, 2.0, 0.5, 1.0);
        assert!(result.abs() < 1e-12);
    }

    // The fitter should recover known parameters from a noiseless LPPLS
    // series with near-zero SSE.
    #[test]
    fn test_fit_argmin_recovers_synthetic_lppls() {
        let (tc_true, m_true, w_true) = (130.0, 0.5, 8.0);
        let (a_true, b_true, c1_true, c2_true) = (1.0, -0.02, 0.002, -0.002);

        let time: Vec<f64> = (0..120).map(|t| t as f64).collect();
        let price: Vec<f64> = time
            .iter()
            .map(|&t| {
                let dt: f64 = tc_true - t;
                a_true
                    + dt.powf(m_true)
                        * (b_true
                            + c1_true * (w_true * dt.ln()).cos()
                            + c2_true * (w_true * dt.ln()).sin())
            })
            .collect();

        let (tc, m, w, _a, b, _c1, _c2, cost) =
            fit_argmin(&time, &price).expect("fit should succeed");

        assert!(cost < 1e-3, "cost too high: {}", cost);
        assert!((tc - tc_true).abs() < 5.0, "tc off: {}", tc);
        assert!((m - m_true).abs() < 0.1, "m off: {}", m);
        assert!((w - w_true).abs() < 0.5, "w off: {}", w);
        assert!(b < 0.0, "b should be negative for a positive bubble: {}", b);
    }

    #[test]
    fn test_compute_residual_indicator() {
        // b = c1 = c2 = 0 makes the fitted value equal a (= 0), so the
        // residual is exactly p2: a ramp whose |eps| is always the running
        // max (eps_norm = 1 after burn-in), then a final dip to -0.12
        // against a running max of 0.24 (eps_norm = -0.5).
        let n = 25;
        let dates: Vec<i32> = (0..n as i32).collect();
        let mut p2: Vec<f64> = (0..n).map(|i| 0.01 * (i as f64 + 1.0)).collect();
        p2[24] = -0.12;
        let zeros = vec![0.0; n];
        let df = polars::df!(
            "t2_d" => dates,
            "t2" => vec![100.0; n],
            "p2" => p2,
            "tc" => vec![130.0; n],
            "m" => vec![0.5; n],
            "w" => vec![8.0; n],
            "a" => zeros.clone(),
            "b" => zeros.clone(),
            "c1" => zeros.clone(),
            "c2" => zeros,
        )
        .unwrap()
        .lazy()
        .with_column(col("t2_d").cast(DataType::Date))
        .collect()
        .unwrap();

        let out = compute_residual_indicator(&df).unwrap();
        let e = out.column("eps_norm").unwrap().f64().unwrap();
        assert_eq!(e.get(5).unwrap(), 0.0, "burn-in should zero early values");
        assert!((e.get(21).unwrap() - 1.0).abs() < 1e-9);
        assert!((e.get(24).unwrap() + 0.5).abs() < 1e-9);
    }

    // Too-short or misaligned inputs must error rather than return junk.
    #[test]
    fn test_fit_argmin_rejects_bad_input() {
        let time: Vec<f64> = (0..5).map(|t| t as f64).collect();
        let price = vec![1.0; 5];
        assert!(fit_argmin(&time, &price).is_err());

        let time: Vec<f64> = (0..20).map(|t| t as f64).collect();
        let price = vec![1.0; 19];
        assert!(fit_argmin(&time, &price).is_err());
    }
}
