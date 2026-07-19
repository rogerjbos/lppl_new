use chrono::{Duration, NaiveDate};
use clickhouse::{Client, Row};
use csv::WriterBuilder;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::{env, error::Error as StdError, fmt::Debug}; //, process::Command
use tokio::time;

// Add this enum above the client functions
pub enum ChConnectionType {
    Local,
    Remote,
}

#[derive(Debug, Row, Serialize, Deserialize)]
struct OHLCV {
    date: String,
    ticker: String,
    universe: String,
    open: Option<f64>,
    high: Option<f64>,
    low: Option<f64>,
    close: Option<f64>,
    volume: Option<f64>,
}

// Helper struct for get_universe_tickers
#[derive(Row, Deserialize, Debug)]
struct TickerRow {
    _ticker: String,
}

pub async fn write_price_file(univ: String, production: bool) -> Result<(), Box<dyn StdError>> {
    let user_path = match env::var("CLICKHOUSE_USER_PATH") {
        Ok(path) => path,
        Err(_) => String::from("/srv"),
    };
    let folder = if production { "production" } else { "testing" };
    let filename = format!(
        "{}/rust_home/lppl_new/data/{}/{}.csv",
        user_path.to_string(),
        folder.to_string(),
        univ
    );

    // Get the list of tickers in the universe that are already pre-filtered for validity
    let tickers = get_universe_tickers(&univ, production).await?;

    // Process in chunks of 25 tickers (reduced to avoid server memory limit)
    let chunk_size = 25;
    let ticker_chunks: Vec<Vec<String>> = tickers
        .chunks(chunk_size)
        .map(|chunk| chunk.to_vec())
        .collect();

    println!(
        "{} Processing {} chunks for {} tickers",
        univ,
        ticker_chunks.len(),
        tickers.len()
    );

    // Get a client connection once
    let client = get_ch_client(ChConnectionType::Local).await?;

    // Create the final CSV file and writer once
    let file = File::create(&filename)?;
    let mut wtr = WriterBuilder::new().has_headers(false).from_writer(file);

    // Write the header record once
    wtr.write_record(&[
        "Date", "Ticker", "Universe", "Open", "High", "Low", "Close", "Volume",
    ])?;

    // Process each chunk
    for (i, chunk) in ticker_chunks.iter().enumerate() {
        // Join the ticker list into a quoted, comma-separated string for SQL IN clause
        let ticker_list = chunk
            .iter()
            .map(|t| format!("'{}'", t))
            .collect::<Vec<_>>()
            .join(",");

        println!("Processing chunk {} with {} tickers", i, chunk.len());

        // Retry logic - try up to 3 times
        let mut attempts = 0;
        const MAX_ATTEMPTS: usize = 3;
        let mut chunk_success = false;

        while !chunk_success && attempts < MAX_ATTEMPTS {
            attempts += 1;

            let query = if production && univ == "Crypto" {
                format!("WITH primary_quotes AS (
                    -- Step 1: Find the quote currency with the max volume for each (baseCurrency, date)
                    SELECT
                        baseCurrency,
                        date,
                        quoteCurrency,
                        open,
                        high,
                        low,
                        close,
                        volume,
                        volumeNotional,
                        tradesDone,
                        ticker
                    FROM (
                        SELECT
                            baseCurrency,
                            date,
                            quoteCurrency,
                            open,
                            high,
                            low,
                            close,
                            volume,
                            volumeNotional,
                            tradesDone,
                            ticker,
                            row_number() OVER (PARTITION BY baseCurrency, date ORDER BY volume DESC) as volume_rank
                        FROM crypto
                    ) WHERE volume_rank = 1
                ),
                univ AS (
                    -- Step 2: Your original universe selection logic, now applied to the de-duplicated data
                    SELECT
                        baseCurrency ticker,
                        max(formatDateTime(date, '%Y-%m-%d')) maxdate
                    FROM primary_quotes -- Use the de-duplicated data here
                    WHERE baseCurrency IN ({})
                    GROUP BY ticker
                    HAVING count(date) > 500 AND COUNT(*) * 2 - COUNT(high) - COUNT(low) = 0
                )
                -- Step 3: Final selection, also joining against the de-duplicated data
                SELECT
                    formatDateTime(date, '%Y-%m-%d') Date,
                    u.ticker Ticker,
                    'Crypto' as Universe,
                    pq.open AS Open,
                    pq.high AS High,
                    pq.low AS Low,
                    pq.close AS Close,
                    pq.volume AS Volume
                FROM primary_quotes pq -- Use the de-duplicated data here
                INNER JOIN univ u ON u.ticker = pq.baseCurrency
                WHERE pq.date > '2020-01-01' AND NOT match(u.ticker, '\\d[ls]$')
                ORDER BY Ticker, Date", ticker_list)
            } else if production && univ != "Crypto" {
                format!(
                    "WITH
                    max_usd_date AS (
                        SELECT max(date(formatDateTime(date, '%Y-%m-%d'))) AS max_date
                        FROM usd
                    ),
                    mdate AS (
                        SELECT symbol, max(date) AS maxdate
                        FROM usd p
                        WHERE symbol IN ({})
                        group by symbol
                        having count(date) >= 130 and COUNT(*) * 2 - COUNT(adjHigh) - COUNT(adjLow) = 0
                    )
                    SELECT toString(date) Date
                    , p.symbol AS Ticker
                    , '{univ}' AS Universe
                    , round(adjOpen, 2) AS Open
                    , round(adjHigh, 2) AS High
                    , round(adjLow, 2) AS Low
                    , round(adjClose, 2) AS Close
                    , round(adjVolume, 2) AS Volume
                    FROM usd p FINAL
                    INNER JOIN mdate m ON m.symbol = p.symbol
                    CROSS JOIN max_usd_date mu
                    WHERE p.date >= subtractDays(now(), 365)
                    AND m.maxdate = mu.max_date
                    order by Ticker, Date",
                    ticker_list
                )
            } else if !production && univ == "Crypto" {
                format!(
                    "WITH primary_quotes AS (
                    -- Step 1: Find the quote currency with the max volume for each (baseCurrency, date)
                    SELECT
                        baseCurrency,
                        date,
                        quoteCurrency,
                        open,
                        high,
                        low,
                        close,
                        volume,
                        volumeNotional,
                        tradesDone,
                        ticker
                    FROM (
                        SELECT
                            baseCurrency,
                            date,
                            quoteCurrency,
                            open,
                            high,
                            low,
                            close,
                            volume,
                            volumeNotional,
                            tradesDone,
                            ticker,
                            row_number() OVER (PARTITION BY baseCurrency, date ORDER BY volume DESC) as volume_rank
                        FROM crypto
                    ) WHERE volume_rank = 1
                ),
                univ AS (
                    -- Step 2: Your original universe selection logic, now applied to the de-duplicated data
                    SELECT
                        baseCurrency ticker,
                        max(formatDateTime(date, '%Y-%m-%d')) maxdate
                    FROM primary_quotes
                    WHERE baseCurrency IN ({})
                    GROUP BY ticker
                    HAVING count(date) > 500 AND COUNT(*) * 2 - COUNT(high) - COUNT(low) = 0
                )
                -- Step 3: Final selection, also joining against the de-duplicated data
                SELECT
                    formatDateTime(pq.date, '%Y-%m-%d') Date,
                    u.ticker Ticker,
                    'Crypto' as Universe,
                    pq.open AS Open,
                    pq.high AS High,
                    pq.low AS Low,
                    pq.close AS Close,
                    pq.volume AS Volume
                FROM primary_quotes pq -- Use the de-duplicated data here
                INNER JOIN univ u ON u.ticker = pq.baseCurrency
                WHERE pq.date > '2020-01-01' AND NOT match(u.ticker, '\\d[ls]$')
                ORDER BY Ticker, Date",
                    ticker_list
                )
            } else {
                format!(
                    "WITH
                    max_usd_date AS (
                        SELECT max(date(formatDateTime(date, '%Y-%m-%d'))) AS max_date
                        FROM usd
                    ),
                    mdate AS (
                        SELECT symbol, max(date(formatDateTime(date, '%Y-%m-%d'))) AS maxdate
                        FROM usd p
                        WHERE symbol IN ({})
                        group by symbol
                        having count(date) >= 250 and COUNT(*) * 2 - COUNT(adjHigh) - COUNT(adjLow) = 0
                    )
                    SELECT toString(date(formatDateTime(p.date, '%Y-%m-%d'))) Date
                    , p.symbol AS Ticker
                    , '{univ}' AS Universe
                    , round(adjOpen, 2) AS Open
                    , round(adjHigh, 2) AS High
                    , round(adjLow, 2) AS Low
                    , round(adjClose, 2) AS Close
                    , round(adjVolume, 2) AS Volume
                    FROM usd p FINAL
                    INNER JOIN mdate m ON m.symbol = p.symbol
                    CROSS JOIN max_usd_date mu
                    WHERE p.date >= subtractDays(now(), 365)
                    AND m.maxdate = mu.max_date
                    order by Ticker, Date",
                    ticker_list
                )
            };
            println!(
                "Executing query for chunk {}/{} (attempt {})",
                i + 1,
                ticker_chunks.len(),
                attempts
            );

            // Execute the query and write to the single CSV file
            let mut cursor = client.query(&query).fetch::<OHLCV>()?;
            let mut wrote_row = false;

            while let Some(row) = cursor.next().await? {
                wtr.serialize(row)?;
                wrote_row = true;
            }

            if wrote_row {
                chunk_success = true;
            } else {
                println!("No rows returned for chunk {} on attempt {}", i, attempts);
            }

            if !chunk_success && attempts < MAX_ATTEMPTS {
                println!("Retrying in 5 seconds...");
                std::thread::sleep(std::time::Duration::from_secs(5));
            } else if !chunk_success && attempts == MAX_ATTEMPTS {
                eprintln!("All attempts failed for chunk {}", i);
            }
        }
    }
    wtr.flush()?;
    println!("Successfully created {}", filename);
    Ok(())
}

// Function to get the list of tickers in a universe
async fn get_universe_tickers(
    univ: &str,
    _production: bool,
) -> Result<Vec<String>, Box<dyn StdError>> {
    let client = get_ch_client(ChConnectionType::Local).await?;

    let query = if univ == "Crypto" {
        "SELECT DISTINCT baseCurrency FROM crypto ORDER BY baseCurrency".to_string()
    } else {
        format!(
            "SELECT DISTINCT Ticker FROM univ WHERE batch = '{}' ORDER BY Ticker",
            univ
        )
    };

    let tickers: Vec<String> = client.query(&query).fetch_all().await?;
    Ok(tickers)
}

fn read_env_var(key: &str) -> String {
    env::var(key).unwrap_or_else(|_| panic!("{key} env variable should be set"))
}

pub async fn get_ch_client(connection_type: ChConnectionType) -> Result<Client, Box<dyn StdError>> {
    let (url, user, password, database, conn_type_str) = match connection_type {
        ChConnectionType::Local => {
            let host = "192.168.86.46";
            (
                format!("http://{}:8123", host),
                "roger".to_string(),
                read_env_var("PG"),
                "tiingo".to_string(),
                "Local",
            )
        }
        ChConnectionType::Remote => {
            let host = "192.168.86.56";
            (
                format!("http://{}:8123", host),
                "roger".to_string(),
                read_env_var("PG"),
                "tiingo".to_string(),
                "Remote",
            )
        }
    };

    let client = Client::default()
        .with_url(url)
        .with_user(user)
        .with_password(password)
        .with_database(database);

    let query_result = client.query("SELECT version()").fetch_one::<String>().await;

    match query_result {
        Ok(version) => {
            println!(
                "Successfully connected to ClickHouse {}. Server version: {}",
                conn_type_str, version
            );
            Ok(client)
        }
        Err(e) => {
            println!("Failed to connect to ClickHouse {}: {:?}", conn_type_str, e);
            Err(Box::new(e))
        }
    }
}

#[derive(Clone, Debug, Row, Serialize)]
struct Score {
    date: String,
    universe: String,
    ticker: String,
    side: i64,
    risk_reward: f64,
    expectancy: f64,
    profit_factor: f64,
    pos_conf: f64,
    neg_conf: f64,
}

pub async fn insert_score_dataframe(df: DataFrame) -> Result<(), Box<dyn StdError>> {
    // Create both clients
    let client_local = get_ch_client(ChConnectionType::Local).await?;
    let client_remote = get_ch_client(ChConnectionType::Remote).await?;

    // Extract column data once
    let date_column = df.column("date")?.date()?;
    let universe_column = df.column("universe")?.str()?;
    let ticker_column = df.column("ticker")?.str()?;
    let side_column = df.column("side")?.i64()?;
    let risk_reward_column = df.column("risk_reward")?.f64()?;
    let expectancy_column = df.column("expectancy")?.f64()?;
    let profit_factor_column = df.column("profit_factor")?.f64()?;
    let pos_conf_column = df.column("pos_conf")?.f64()?;
    let neg_conf_column = df.column("neg_conf")?.f64()?;

    // Create a vector of (client, name, use_binary) tuples to process
    let clients = vec![
        (client_local, "local", true),
        (client_remote, "remote", false), // Use SQL INSERT for remote due to version incompatibility
    ];

    for (client, location, use_binary) in clients {
        let result = async {
            if use_binary {
                // Use binary format for local (faster)
                let batch_size = 1000;
                for batch_start in (0..df.height()).step_by(batch_size) {
                    let batch_end = (batch_start + batch_size).min(df.height());
                    let mut insert = client.insert("lppl_score")?;

                    for i in batch_start..batch_end {
                        let date_days = date_column.get(i).unwrap();
                        let naive_date = NaiveDate::from_ymd_opt(1970, 1, 1).expect("Invalid base date")
                            + Duration::days(date_days as i64);
                        let date_str = naive_date.to_string();

                        let row = Score {
                            date: date_str,
                            universe: universe_column.get(i).unwrap().to_string(),
                            ticker: ticker_column.get(i).unwrap().to_string(),
                            side: side_column.get(i).unwrap(),
                            risk_reward: risk_reward_column.get(i).unwrap(),
                            expectancy: expectancy_column.get(i).unwrap(),
                            profit_factor: profit_factor_column.get(i).unwrap(),
                            pos_conf: pos_conf_column.get(i).unwrap(),
                            neg_conf: neg_conf_column.get(i).unwrap(),
                        };
                        insert.write(&row).await?;
                    }
                }
            } else {
                // Use SQL VALUES format for remote (more compatible across versions)
                let batch_size = 50;
                for batch_start in (0..df.height()).step_by(batch_size) {
                    let batch_end = (batch_start + batch_size).min(df.height());

                    let mut values = Vec::new();
                    for i in batch_start..batch_end {
                        let date_days = date_column.get(i).unwrap();
                        let naive_date = NaiveDate::from_ymd_opt(1970, 1, 1).expect("Invalid base date")
                            + Duration::days(date_days as i64);
                        let date_str = naive_date.to_string();

                        let universe = universe_column.get(i).unwrap();
                        let ticker = ticker_column.get(i).unwrap().replace("'", "''"); // Escape single quotes
                        let side = side_column.get(i).unwrap();
                        let risk_reward = risk_reward_column.get(i).unwrap();
                        let expectancy = expectancy_column.get(i).unwrap();
                        let profit_factor = profit_factor_column.get(i).unwrap();
                        let pos_conf = pos_conf_column.get(i).unwrap();
                        let neg_conf = neg_conf_column.get(i).unwrap();

                        values.push(format!(
                            "('{}', '{}', '{}', {}, {}, {}, {}, {}, {})",
                            date_str, universe, ticker, side, risk_reward, expectancy,
                            profit_factor, pos_conf, neg_conf
                        ));
                    }

                    let query = format!(
                        "INSERT INTO lppl_score (date, universe, ticker, side, risk_reward, expectancy, \
                         profit_factor, pos_conf, neg_conf) VALUES {}",
                        values.join(", ")
                    );

                    client.query(&query).execute().await?;

                    // Progress indicator
                    if batch_end % 100 == 0 || batch_end == df.height() {
                        println!("Progress {}: {}/{} rows", location, batch_end, df.height());
                    }

                    // Small delay between batches
                    time::sleep(time::Duration::from_millis(100)).await;
                }
            }
            Ok::<(), Box<dyn StdError>>(())
        }
        .await;

        match result {
            Ok(_) => println!(
                "Successfully inserted {} rows into ClickHouse {}",
                df.height(),
                location
            ),
            Err(e) => eprintln!(
                "Failed to insert rows into ClickHouse {}: {:?}",
                location, e
            ),
        }
    }

    Ok(())
}
