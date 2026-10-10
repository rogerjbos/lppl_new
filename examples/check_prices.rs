// Data-quality report for a price CSV: what the data-break screen in
// read_price_file keeps per ticker. Usage:
// cargo run --release --example check_prices data/testing/Micro1.csv
use lppl_new::read_price_file;
use polars::prelude::*;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = env::args().nth(1).expect("usage: check_prices <price.csv>");
    let raw = LazyCsvReader::new(file.clone())
        .with_has_header(true)
        .finish()?
        .group_by([col("Ticker")])
        .agg([col("Close").count().cast(DataType::Int64).alias("raw_rows")]);
    let kept = read_price_file(file)
        .await?
        .group_by([col("Ticker")])
        .agg([
            col("Close").count().cast(DataType::Int64).alias("kept_rows"),
            col("Date").min().alias("first"),
            col("Date").max().alias("last"),
            col("Close").min().alias("min_close"),
        ]);
    let df = raw
        .join(kept, [col("Ticker")], [col("Ticker")], JoinArgs::new(JoinType::Left))
        .with_column(col("kept_rows").fill_null(lit(0i64)))
        .with_column((col("raw_rows") - col("kept_rows")).alias("dropped"))
        .sort(["dropped"], SortMultipleOptions::default().with_order_descending(true))
        .collect()?;
    let trimmed = df.clone().lazy().filter(col("dropped").gt(lit(0i64))).collect()?;
    let lost = df.clone().lazy().filter(col("kept_rows").lt(lit(130i64))).collect()?;
    println!(
        "tickers {}, trimmed {}, left with <130 rows {}",
        df.height(),
        trimmed.height(),
        lost.height()
    );
    env::set_var("POLARS_FMT_MAX_ROWS", "25");
    println!("{}", trimmed.head(Some(25)));

    // Per-ticker scaling check: filtering by ticker first (as fits_helper
    // does) must give a 0..1 range for every ticker.
    let lf = read_price_file(env::args().nth(1).unwrap()).await?;
    for t in df.column("Ticker")?.str()?.into_iter().flatten().take(3) {
        let r = lf
            .clone()
            .filter(col("Ticker").eq(lit(t)))
            .select([
                col("scaled_price_ln").min().alias("min"),
                col("scaled_price_ln").max().alias("max"),
            ])
            .collect()?;
        println!("{} scaled_price_ln range: {}", t, r);
    }
    Ok(())
}
