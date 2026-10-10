// Per-ticker, per-strategy backtest aggregates from a parquet folder, written
// to CSV for outlier / robustness analysis of the summary metrics. Usage:
// cargo run --release --example dump_stats <dir> <out.csv>
use polars::prelude::*;
use std::env;
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = env::args().nth(1).unwrap_or_else(|| "output/testing".to_string());
    let out = env::args().nth(2).unwrap_or_else(|| "ticker_stats.csv".to_string());
    let firsts = [
        "trades", "wins", "losses", "gain_sum", "loss_sum", "ret_sq_sum", "max_gain", "max_loss",
        "buys", "sells", "exit_signal", "exit_stop", "exit_tc", "exit_conf", "exit_hold",
        "exit_eod",
    ];
    let mut df = LazyFrame::scan_parquet(format!("{}/*.parquet", dir), ScanArgsParquet::default())?
        .group_by([col("ticker"), col("universe"), col("strategy")])
        .agg(firsts.iter().map(|c| col(*c).first()).collect::<Vec<_>>())
        .filter(col("trades").gt(lit(0)))
        .sort(["strategy", "universe", "ticker"], SortMultipleOptions::default())
        .collect()?;
    let mut file = File::create(&out)?;
    CsvWriter::new(&mut file).finish(&mut df)?;
    println!("wrote {} rows to {}", df.height(), out);
    Ok(())
}
