// Calibrate the absolute fit-quality gate (FIT_R2_MIN) from existing fit
// files: recompute each window's R^2 = 1 - SSE/SST and print its
// distribution. Usage: cargo run --release --example calibrate_r2 [production|testing]
use lppl_new::date_to_ce;
use chrono::NaiveDate;
use std::collections::HashMap;
use std::env;
use std::fs;

#[derive(Default)]
struct Series {
    time: Vec<f64>,
    ln_close: Vec<f64>,
}

fn min_max(v: &[f64]) -> (f64, f64) {
    let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (lo, hi)
}

fn fitted(t: f64, tc: f64, m: f64, w: f64, a: f64, b: f64, c1: f64, c2: f64) -> f64 {
    let dt = (tc - t).abs().max(1e-8);
    let l = dt.ln();
    a + dt.powf(m) * (b + c1 * (w * l).cos() + c2 * (w * l).sin())
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let folder = env::args().nth(1).unwrap_or_else(|| "production".to_string());
    let root = "/Users/rogerbos/rust_home/lppl_new";

    // r2 values, keyed by window-length bucket, plus overall
    let mut all: Vec<f64> = Vec::new();
    let mut by_len: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut basis_ticker = 0usize;
    let mut basis_universe = 0usize;

    for entry in fs::read_dir(format!("{}/data/{}", root, folder))? {
        let path = entry?.path();
        let univ = path.file_stem().unwrap().to_str().unwrap().to_string();
        if univ == "Crypto" {
            continue;
        }
        // load the universe price file: ticker -> (time, ln close)
        let mut rdr = csv::Reader::from_path(&path)?;
        let mut series: HashMap<String, Series> = HashMap::new();
        let mut univ_ln: Vec<f64> = Vec::new();
        for rec in rdr.records() {
            let rec = rec?;
            let close: f64 = match rec[6].parse() {
                Ok(c) if c > 0.0 => c,
                _ => continue,
            };
            let date = NaiveDate::parse_from_str(&rec[0], "%Y-%m-%d")?;
            let s = series.entry(rec[1].to_string()).or_default();
            s.time.push(date_to_ce(date));
            s.ln_close.push(close.ln());
            univ_ln.push(close.ln());
        }
        let (ulo, uhi) = min_max(&univ_ln);

        for (ticker, s) in &series {
            let fname = format!("{}/fit/{}/fit_{}_{}.csv", root, folder, univ, ticker);
            let Ok(mut frdr) = csv::Reader::from_path(&fname) else {
                continue;
            };
            let (tlo, thi) = min_max(&s.ln_close);
            for rec in frdr.records() {
                let rec = rec?;
                let f = |i: usize| -> f64 { rec[i].parse().unwrap_or(f64::NAN) };
                let (tc, m, w, a, b, c1, c2) = (f(1), f(2), f(3), f(4), f(5), f(7), f(8));
                let (t1, t2, sse) = (f(11), f(12), f(16));
                if !sse.is_finite() {
                    continue;
                }
                let idx: Vec<usize> = (0..s.time.len())
                    .filter(|&i| s.time[i] >= t1 && s.time[i] <= t2)
                    .collect();
                if idx.len() < 10 {
                    continue;
                }
                // Which min-max basis reproduces the recorded SSE?
                let eval = |lo: f64, hi: f64| -> (f64, f64) {
                    let p: Vec<f64> = idx.iter().map(|&i| (s.ln_close[i] - lo) / (hi - lo)).collect();
                    let mean = p.iter().sum::<f64>() / p.len() as f64;
                    let sst: f64 = p.iter().map(|x| (x - mean).powi(2)).sum();
                    let sse_re: f64 = idx
                        .iter()
                        .zip(p.iter())
                        .map(|(&i, &pv)| (pv - fitted(s.time[i], tc, m, w, a, b, c1, c2)).powi(2))
                        .sum();
                    (sst, sse_re)
                };
                let (sst_t, sse_t) = eval(tlo, thi);
                let (sst_u, sse_u) = eval(ulo, uhi);
                let sst = if (sse_t - sse).abs() <= (sse_u - sse).abs() {
                    basis_ticker += 1;
                    sst_t
                } else {
                    basis_universe += 1;
                    sst_u
                };
                if sst <= 0.0 {
                    continue;
                }
                let r2 = 1.0 - sse / sst;
                all.push(r2);
                by_len.entry(idx.len() / 20 * 20).or_default().push(r2);
            }
        }
        println!("{} done ({} fits so far)", univ, all.len());
    }

    println!(
        "scaling basis matched: per-ticker {} / universe-wide {}",
        basis_ticker, basis_universe
    );
    all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("R^2 quantiles over {} fits:", all.len());
    for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90] {
        println!("  q{:>4.2}: {:.3}", q, quantile(&all, q));
    }
    println!("pass rate by threshold:");
    for thr in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95] {
        let pass = all.iter().filter(|&&r| r >= thr).count() as f64 / all.len() as f64;
        println!("  r2 >= {:.2}: {:.1}%", thr, pass * 100.0);
    }
    let mut lens: Vec<_> = by_len.keys().cloned().collect();
    lens.sort();
    println!("median / q10 R^2 by window length (rows):");
    for l in lens {
        let mut v = by_len.remove(&l).unwrap();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "  {:>3}-{:<3}: n={:>6} median={:.3} q10={:.3}",
            l,
            l + 19,
            v.len(),
            quantile(&v, 0.5),
            quantile(&v, 0.1)
        );
    }
    Ok(())
}
