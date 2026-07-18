// End-to-end sanity check of the LPPLS pipeline on the canonical dot-com
// bubble data (NASDAQ, ending at the 2000-03-10 peak). Expect pos_conf to
// be clearly positive in the weeks before the peak, with fitted tc dates
// clustering around spring 2000.
//
// Run with: cargo run --release --example dotcom
use lppl_new::{compute_indicators, compute_nested_fits, load_example_data};
use polars::prelude::*;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let df = load_example_data()?.collect()?;

    let time: Vec<f64> = df.column("time")?.f64()?.into_no_null_iter().collect();
    let price: Vec<f64> = df.column("price")?.f64()?.into_no_null_iter().collect();

    // Last ~400 trading days (mid-1998 through the 2000-03-10 peak)
    let n = time.len();
    let start = n - 400;
    let time = &time[start..];
    let price = &price[start..];

    let fits = compute_nested_fits(time, price, 120, 30, 5, 5)?;

    println!(
        "Sample of fits near the peak:\n{}",
        fits.select(["t2_d", "tc_d", "m", "w", "b", "D", "O", "best_cost"])?
            .tail(Some(10))
    );

    let ind = compute_indicators(fits)?
        .sort(["t2_d"], SortMultipleOptions::default())?
        .select(["t2_d", "pos_conf", "neg_conf", "price"])?;

    println!("Confidence indicator (last 20 dates):\n{}", ind.tail(Some(20)));
    Ok(())
}
