// Ungated per-strategy signal/trade counts from a backtest parquet folder,
// for checking signal rates on small runs. Usage:
// cargo run --release --example trade_counts <dir>
use polars::prelude::*;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = env::args().nth(1).unwrap_or_else(|| "output/testing".to_string());
    let df = LazyFrame::scan_parquet(format!("{}/*.parquet", dir), ScanArgsParquet::default())?
        .group_by([col("strategy")])
        .agg([
            col("buys").sum().alias("buy_sigs"),
            col("sells").sum().alias("sell_sigs"),
            col("trades").sum().alias("trades"),
            (col("trades").gt(lit(0))).sum().alias("tickers_traded"),
            ((col("gain_sum") - col("loss_sum")).sum() / col("trades").sum().cast(DataType::Float64))
                .alias("mean_ret"),
            col("exit_signal").sum().alias("x_sig"),
            col("exit_stop").sum().alias("x_stop"),
            col("exit_tc").sum().alias("x_tc"),
            col("exit_conf").sum().alias("x_conf"),
            col("exit_hold").sum().alias("x_hold"),
            col("exit_eod").sum().alias("x_eod"),
        ])
        .sort(["trades"], SortMultipleOptions::default().with_order_descending(true))
        .collect()?;
    env::set_var("POLARS_FMT_MAX_ROWS", "30");
    env::set_var("POLARS_FMT_MAX_COLS", "20");
    println!("{}", df.head(Some(30)));
    println!("strategies with any trade: {}", df.lazy().filter(col("trades").gt(lit(0))).collect()?.height());
    Ok(())
}
