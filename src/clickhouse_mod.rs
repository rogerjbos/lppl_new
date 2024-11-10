use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::{env, error::Error as StdError, fmt::Debug, process::Command};
use clickhouse::{Client, Row};
use chrono::{NaiveDate,  Duration};


#[derive(Debug, Row, Serialize, Deserialize)]
struct OHLCV {
    date: String,
    ticker: String,
    universe: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64
}


pub async fn write_price_file(univ: String, production: bool) -> Result<(), Box<dyn StdError>> {
 
    let user_path = match env::var("CLICKHOUSE_USER_PATH") {
        Ok(path) => path,
        Err(_) => String::from("/srv"),
    };
    let folder = if production { "production" } else { "testing" };
    let filename = format!("{}/rust_home/lppl_new/data/{}/{}.csv", user_path.to_string(), folder.to_string(), univ);
 
    let query = if production && univ == "Crypto" { "WITH univ AS (
        SELECT baseCurrency ticker, max(date) maxdate
        FROM crypto
        group by ticker
        having count(date) > 250 and COUNT(*) * 2 - COUNT(high) - COUNT(low) = 0
        )
        SELECT date(p.date) Date, u.ticker Ticker, 'Crypto' as Universe,
        open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume
        FROM crypto p
        INNER JOIN univ u
        ON u.ticker = p.baseCurrency
        WHERE p.date >= subtractDays(now(), 250)
        and maxdate IN (select max(date) from crypto)
        order by ticker, date".to_string()
    } else if production && univ != "Crypto"{ format!("WITH mdate AS (
        SELECT symbol, max(date(date)) AS maxdate
        FROM usd p
        INNER JOIN univ u
        ON p.symbol = u.Ticker and u.batch ='{univ}'
        group by symbol
        having count(date) >= 250 and COUNT(*) * 2 - COUNT(adjHigh) - COUNT(adjLow) = 0
        )
        SELECT date(p.date) Date
        , symbol AS Ticker
        , '{univ}' AS Universe
        , round(adjOpen, 2) AS Open
        , round(adjHigh, 2) AS High
        , round(adjLow, 2) AS Low
        , round(adjClose, 2) AS Close
        , round(adjVolume, 2) AS Volume
        FROM usd p
        INNER JOIN mdate m
        ON m.symbol = p.symbol
        WHERE p.date >= subtractDays(now(), 365)
        and m.maxdate IN (select max(date(date)) from usd)
        order by Ticker, date")
    } else if !production && univ == "Crypto" { "WITH univ AS (
        SELECT baseCurrency ticker, max(date) maxdate
        FROM crypto
        group by ticker
        having count(date) > 500 and COUNT(*) * 2 - COUNT(high) - COUNT(low) = 0
        )
        SELECT date(p.date) Date, u.ticker Ticker, 'Crypto' as Universe,
        open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume
        FROM crypto p
        INNER JOIN univ u
        ON u.ticker = p.baseCurrency
        WHERE date > '2020-01-01'
        order by ticker, date".to_string()
    } else if !production && univ != "Crypto" { format!("WITH mdate AS (
        SELECT symbol, max(date(date)) AS maxdate
        FROM usd p
        INNER JOIN univ u
        ON p.symbol = u.Ticker and u.batch ='{univ}'
        group by symbol
        having count(date) >= 250 and COUNT(*) * 2 - COUNT(adjHigh) - COUNT(adjLow) = 0
        )
        SELECT date(p.date) Date
        , symbol AS Ticker
        , '{univ}' AS Universe
        , round(adjOpen, 2) AS Open
        , round(adjHigh, 2) AS High
        , round(adjLow, 2) AS Low
        , round(adjClose, 2) AS Close
        , round(adjVolume, 2) AS Volume
        FROM usd p
        INNER JOIN mdate m
        ON m.symbol = p.symbol
        WHERE p.date >= subtractDays(now(), 365)
        and m.maxdate IN (select max(date(date)) from usd)
        order by Ticker, date")
    } else {
        panic!("Error: no query match")
    };

    let user = env::var("CLICKHOUSE_USER")?;
    let pw = env::var("CLICKHOUSE_PASSWORD")?;
    let cmd = format!(r#"clickhouse client --host='vdib5n7pan.europe-west4.gcp.clickhouse.cloud' --user='{}' --password='{}' --secure --database=tiingo -q "{}" --format=CSVWithNames > {}"#, user, pw, query, filename.clone());

    let output = Command::new("/bin/sh")
        .arg("-c")
        .arg(&cmd)
        .output()?;
    
    if !output.status.success() {
        eprintln!("Query failed with status: {:?}", output.status);
        eprintln!("stderr: {:?}", String::from_utf8_lossy(&output.stderr));
        return Err("Failed to execute query".into());
    }

    Ok(())
}

fn read_env_var(key: &str) -> String {
    env::var(key).unwrap_or_else(|_| panic!("{key} env variable should be set"))
}

pub async fn get_ch_cloud_client() -> Result<Client, Box<dyn StdError>> {
    let client = Client::default()
        .with_url("https://vdib5n7pan.europe-west4.gcp.clickhouse.cloud")
        .with_user(read_env_var("CLICKHOUSE_USER"))
        .with_password(read_env_var("CLICKHOUSE_CLOUD_PASSWORD"))
        .with_database("tiingo");
    let query_result = client.query("SELECT version()").fetch_one::<String>().await;

    match query_result {
        Ok(version) => {
            println!("Successfully connected to ClickHouse. Server version: {}", version);
            Ok(client)  // Connection is successful
        }
        Err(e) => {
            println!("Failed to connect to ClickHouse: {:?}", e);
            Err(Box::new(e))  // Propagate the error
        }
    }    
}

pub async fn get_ch_client() -> Result<Client, Box<dyn StdError>> {
    let client = Client::default()
        .with_url("http://192.168.86.73:8123")
        .with_user(read_env_var("CLICKHOUSE_USER"))
        .with_password(read_env_var("CLICKHOUSE_PASSWORD"))
        .with_database("tiingo");
    let query_result = client.query("SELECT version()").fetch_one::<String>().await;

    match query_result {
        Ok(version) => {
            println!("Successfully connected to ClickHouse. Server version: {}", version);
            Ok(client)  // Connection is successful
        }
        Err(e) => {
            println!("Failed to connect to ClickHouse: {:?}", e);
            Err(Box::new(e))  // Propagate the error
        }
    }    
}

pub async fn insert_score_dataframe(df: DataFrame) -> Result<(), Box<dyn StdError>> {

    let client = get_ch_client().await?;

    let date_column = df.column("date")?.date()?;
    let universe_column = df.column("universe")?.str()?;
    let ticker_column = df.column("ticker")?.str()?;
    let side_column = df.column("side")?.i64()?;
    let risk_reward_column = df.column("risk_reward")?.f64()?;
    let expectancy_column = df.column("expectancy")?.f64()?;
    let profit_factor_column = df.column("profit_factor")?.f64()?;
    
    let mut insert = client.insert("strategy_score")?;
    for i in 0..df.height() {
        let date_days = date_column.get(i).unwrap();  // Number of days since 1970-01-01
        let naive_date = NaiveDate::from_ymd_opt(1970, 1, 1).expect("date") + Duration::days(date_days as i64);
        let naive_datetime = naive_date.and_hms_opt(0, 0, 0).expect("date").and_utc().timestamp() * 1000;
        // let naive_date = NaiveDate::from_ymd_opt(1970, 1, 1).expect("date") + Duration::days(date_days as i64);
        // let naive_datetime = naive_date.and_hms_opt(0, 0, 0).expect("date").and_utc().timestamp() * 1000;

        let row = Score {
            date: naive_datetime,
            universe: universe_column.get(i).unwrap().to_string(),
            ticker: ticker_column.get(i).unwrap().to_string(),
            side: side_column.get(i).unwrap(),
            risk_reward: risk_reward_column.get(i).unwrap(),
            expectancy: expectancy_column.get(i).unwrap(),
            profit_factor: profit_factor_column.get(i).unwrap(),
        };
        insert.write(&row).await?;
    }
    insert.end().await?;
    Ok(())
}

#[derive(Debug, Row, Serialize, Deserialize)]
struct Score {
    date: i64,
    universe: String,
    ticker: String,
    side: i64,
    risk_reward: f64,
    expectancy: f64,
    profit_factor: f64
}

async fn _create_score_table() -> Result<(), Box<dyn StdError>> {

    let client = get_ch_client().await?;
    let txt: &str = "CREATE OR REPLACE TABLE strategy_score (
        date String,
        universe LowCardinality(String),
        ticker LowCardinality(String),
        side Int64,
        risk_reward Float64,
        expectancy Float64,
        profit_factor Float64 )
    ENGINE = ReplacingMergeTree
    ORDER BY ticker";
    let _ = client.query(&txt).execute().await;

    Ok(())
}
