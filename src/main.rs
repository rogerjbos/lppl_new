#![cfg_attr(debug_assertions, allow(dead_code, unused_imports))]
use crate::backtester::*;
use lppl_new::*;
use polars::prelude::*;
use std::{env, error::Error as StdError};
// use crate::clickhouse_mod::create_score_table;

#[tokio::main]
async fn main() -> Result<(), Box<dyn StdError>> {
    // default params (overwritten by command line args)
    let user_path = match env::var("CLICKHOUSE_USER_PATH") {
        Ok(path) => path,
        Err(_) => String::from("/Users/rogerbos"), // Provide a default path if the environment variable is not set
    };
    let default_path: String = format!("{}/rust_home/lppl_new", user_path);

    let default_production: String = "testing".to_string();
    let default_univ = "Crypto".to_string();
    let batch_size: usize = 10;

    // collect command line args
    let args: Vec<String> = env::args().collect();
    let univ_str: &str = args.get(1).unwrap_or(&default_univ);
    let production_str = args.get(2).unwrap_or(&default_production);
    let path = args.get(3).unwrap_or(&default_path);

    let production = production_str == "production";
    let univ: &[&str] = match univ_str {
        "SC" => &["SC1", "SC2", "SC3", "SC4"],
        "MC" => &["MC1", "MC2"],
        "MC1" => &["MC1"],
        "LC" => &["LC1", "LC2"],
        "Micro" => &["Micro1", "Micro2", "Micro3", "Micro4"],
        "Stocks" => &[
            "SC1", "SC2", "SC3", "SC4", "MC1", "MC2", "LC1", "LC2", "Micro1", "Micro2", "Micro3",
            "Micro4",
        ],
        _ => &["Crypto"],
    };
    let univ_vec: Vec<String> = univ.iter().map(|&s| s.into()).collect();

    let overwrite = true; // DELETE OLD FILES BECAUSE THEY WILL NOT BE OVERWRITTEN
    let run_prices = true;
    let run_fits = true;
    let run_backtests = true;
    let run_performance = true;

    // GENERATE PRICE FILE FOR EACH UNIVERSE
    // saves CSV files for each universe to /data/testing or /data/production
    if run_prices {
        if overwrite {
            // delete_all_files_in_folder(format!("{}/data/{}", path, production_str)).await?;
            for universe in &univ_vec {
                let file_path = format!("{}/data/{}/{}.csv", path, production_str, universe);
                if std::path::Path::new(&file_path).exists() {
                    std::fs::remove_file(file_path)?;
                    println!("Deleted: {}.csv", universe);
                }
            }
        }
        create_price_files(univ_vec.clone(), production.clone()).await?;
        println!("price file done");
    }

    // COMPUTE NESTED FITS FOR EACH UNIVERSE
    // save fit files to /fit/testing or /fit/production
    if run_fits {
        if overwrite {
            let folder = format!("{}/fit/{}", path, production_str);
            // println!("folder: {}", folder);
            delete_all_files_in_folder(folder).await?;
        }
        for u in univ {
            let _ = fits_helper(path.to_string(), u, batch_size, production).await;
        }
        println!("All Fits done");
    }

    // RUN BACKTEST FOR EACH UNIVERSE
    // save parquet files to /output/testing or /output/production
    if run_backtests {
        if overwrite {
            let s = if univ.contains(&"Crypto") {
                "_crypto"
            } else {
                ""
            };
            let folder = format!("{}/output{}/{}", path, s, production_str);
            // println!("folder: {}", folder);
            delete_all_files_in_folder(folder).await?;
        }
        for u in univ {
            let _ = backtest_helper(path.to_string(), u, batch_size, production).await;
        }
        println!("All Backtests done");
    }

    // SHOW AGGREGATED RESULTS BY STRATEGY
    // save csv files to /performance
    if run_performance {
        if overwrite {
            let folder = format!("{}/performance/{}.csv", path, univ_str);
            println!("folder: {}", folder);
            //delete_all_files_in_folder(folder).await?;
        }
        let datetag =
            summary_performance_file((&path).to_string(), production, univ_vec.clone()).await?;
        println!("Done with summary for {}", datetag);

        if production {
            let stocks = if univ.contains(&"Crypto") {
                false
            } else {
                true
            };
            // let _ = create_score_table().await;
            if let Err(e) = score(&datetag, stocks).await {
                eprintln!("Error inserting scores: {}", e);
            }
        } else {
            for u in univ_vec {
                let tag = if u == "Crypto" { "crypto" } else { "stocks" };
                let fname = format!("performance/{}_testing.csv", tag);
                let lf = LazyCsvReader::new(fname).with_has_header(true).finish()?;

                let rr = lf
                    .clone()
                    .filter(col("universe").eq(lit(&*u)))
                    .sort(
                        vec!["risk_reward"],
                        SortMultipleOptions {
                            descending: vec![false],
                            ..Default::default()
                        },
                    )
                    .tail(5)
                    .collect();
                println!("{} risk_reward: {:?}", &u, rr);

                let hr = lf
                    .filter(col("universe").eq(lit(&*u)))
                    .sort(
                        vec!["hit_ratio"],
                        SortMultipleOptions {
                            descending: vec![false],
                            ..Default::default()
                        },
                    )
                    .tail(5)
                    .collect();
                println!("{} hit_ratio: {:?}", &u, hr);
            }
        }
    }
    Ok(())
}
