use argmin::{
    core::{CostFunction, Error, Executor, State},
    solver::neldermead::NelderMead,
};
use nalgebra::{DMatrix, DVector as nalgebraDVector};
use ndarray::{array, Array1};
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use std::error::Error as StdError;

/// Large finite penalty returned instead of INFINITY/NaN so Nelder-Mead can
/// always rank vertices and contract back into the feasible region.
const PENALTY: f64 = 1e10;

/// Number of random restarts per fit; the lowest-SSE solution is kept.
const NUM_RESTARTS: usize = 15;

/// Max Nelder-Mead iterations per restart.
const MAX_ITERS: u64 = 250;

// Search box for the nonlinear parameters. Deliberately wider than the
// qualification filter in compute_indicators (m in [0,1], w in [2,15]) so
// that fits whose true SSE minimum lies outside the filter get found there
// and disqualified, instead of being pinned to the filter boundary and
// fake-qualifying.
const M_SEARCH_MIN: f64 = 1e-3;
const M_SEARCH_MAX: f64 = 2.0;
const W_SEARCH_MIN: f64 = 1.0;
const W_SEARCH_MAX: f64 = 50.0;

/// Cost function over the 3 nonlinear parameters (tc, m, w), with the linear
/// parameters (a, b, c1, c2) subordinated: solved exactly by least squares
/// inside every cost evaluation (Filimonov & Sornette 2013).
#[derive(Debug, Clone)]
struct LpplsCostFunction {
    time: Vec<f64>, // shifted so time[0] == 0
    price: Vec<f64>,
    tc_lo: f64,
    tc_hi: f64,
}

impl CostFunction for LpplsCostFunction {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let tc = param[0];
        let m = param[1];
        let w = param[2];

        // Graded penalty outside the search box gives the simplex a
        // direction back toward feasibility.
        let mut violation = 0.0;
        if tc < self.tc_lo {
            violation += self.tc_lo - tc;
        }
        if tc > self.tc_hi {
            violation += tc - self.tc_hi;
        }
        if m < M_SEARCH_MIN {
            violation += M_SEARCH_MIN - m;
        }
        if m > M_SEARCH_MAX {
            violation += m - M_SEARCH_MAX;
        }
        if w < W_SEARCH_MIN {
            violation += W_SEARCH_MIN - w;
        }
        if w > W_SEARCH_MAX {
            violation += w - W_SEARCH_MAX;
        }
        if violation > 0.0 {
            return Ok(PENALTY * (1.0 + violation));
        }

        let (a, b, c1, c2) = matrix_equation(&self.time, &self.price, tc, m, w);
        let sse = lppls_sse(&self.time, &self.price, tc, m, w, a, b, c1, c2);
        Ok(if sse.is_finite() { sse } else { PENALTY })
    }
}

/// Sum of squared errors of the LPPLS model. Uses |tc - t| (like
/// matrix_equation) clamped away from zero so points near tc cannot
/// produce NaN through ln(0).
pub fn lppls_sse(
    time: &[f64],
    price: &[f64],
    tc: f64,
    m: f64,
    w: f64,
    a: f64,
    b: f64,
    c1: f64,
    c2: f64,
) -> f64 {
    time.iter()
        .zip(price.iter())
        .map(|(&t, &p_obs)| {
            let dt = (tc - t).abs().max(1e-8);
            let log_dt = dt.ln();
            let pow_dt = dt.powf(m);
            let lppls =
                a + pow_dt * (b + c1 * (w * log_dt).cos() + c2 * (w * log_dt).sin());
            (p_obs - lppls).powi(2)
        })
        .sum()
}

// Least-squares solve for the linear parameters (a, b, c1, c2) given the
// nonlinear parameters (tc, m, w).
pub fn matrix_equation(
    time: &[f64],
    price: &[f64],
    tc: f64,
    m: f64,
    w: f64,
) -> (f64, f64, f64, f64) {
    let n = time.len();

    let mut sum_fi = 0.0;
    let mut sum_gi = 0.0;
    let mut sum_hi = 0.0;

    let mut fi_pow_2 = 0.0;
    let mut gi_pow_2 = 0.0;
    let mut hi_pow_2 = 0.0;

    let mut figi = 0.0;
    let mut fihi = 0.0;
    let mut gihi = 0.0;

    let mut sum_yi = 0.0;
    let mut yifi = 0.0;
    let mut yigi = 0.0;
    let mut yihi = 0.0;

    for i in 0..n {
        let dt = (tc - time[i]).abs().max(1e-8);
        let log_dt = dt.ln();

        let fi = dt.powf(m);
        let gi = fi * (w * log_dt).cos();
        let hi = fi * (w * log_dt).sin();
        let yi = price[i];

        sum_fi += fi;
        sum_gi += gi;
        sum_hi += hi;

        fi_pow_2 += fi * fi;
        gi_pow_2 += gi * gi;
        hi_pow_2 += hi * hi;

        figi += fi * gi;
        fihi += fi * hi;
        gihi += gi * hi;

        sum_yi += yi;
        yifi += yi * fi;
        yigi += yi * gi;
        yihi += yi * hi;
    }

    let matrix_1 = DMatrix::from_row_slice(
        4,
        4,
        &[
            n as f64, sum_fi, sum_gi, sum_hi, sum_fi, fi_pow_2, figi, fihi, sum_gi, figi, gi_pow_2,
            gihi, sum_hi, fihi, gihi, hi_pow_2,
        ],
    );

    let matrix_2 = nalgebraDVector::from_row_slice(&[sum_yi, yifi, yigi, yihi]);

    if let Some(solution) = matrix_1.lu().solve(&matrix_2) {
        let a = solution[0];
        let b = solution[1];
        let c1 = solution[2];
        let c2 = solution[3];

        (a, b, c1, c2)
    } else {
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
    }
}

// Fit the LPPLS model: Nelder-Mead over (tc, m, w) with subordinated linear
// parameters, run NUM_RESTARTS times from random seeds, keeping the best fit.
// Time is rescaled so the window starts at 0, which keeps tc on the same
// scale as m and w; tc is shifted back before returning.
pub fn fit_argmin(
    time: &[f64],
    price: &[f64],
) -> Result<(f64, f64, f64, f64, f64, f64, f64, f64), Box<dyn StdError>> {
    if time.len() < 10 || time.len() != price.len() {
        return Err("fit_argmin: need at least 10 aligned observations".into());
    }

    let t0 = time[0];
    let shifted: Vec<f64> = time.iter().map(|t| t - t0).collect();
    let t2s = *shifted.last().unwrap();
    let span = t2s;

    // Same tc range as the tc_in_range qualification filter in
    // compute_indicators, expressed in shifted time.
    let tc_lo = (t2s - 60.0).max(t2s - 0.5 * span);
    let tc_hi = (t2s + 252.0).min(t2s + 0.5 * span);

    let cost = LpplsCostFunction {
        time: shifted.clone(),
        price: price.to_vec(),
        tc_lo,
        tc_hi,
    };

    let mut rng = thread_rng();
    let tc_between = Uniform::from(tc_lo..=tc_hi);
    let m_between = Uniform::from(0.01..=0.99);
    let w_between = Uniform::from(2.0..=15.0);

    // A fit this good (RMSE ~0.1% of the scaled price range) ends the
    // restart loop early.
    let early_exit_sse = 1e-6 * time.len() as f64;

    let mut best: Option<(f64, Array1<f64>)> = None;

    for _ in 0..NUM_RESTARTS {
        let tc0 = tc_between.sample(&mut rng);
        let m0 = m_between.sample(&mut rng);
        let w0 = w_between.sample(&mut rng);
        let init = array![tc0, m0, w0];

        // Non-degenerate simplex: 4 vertices for 3 parameters, each
        // perturbed along one axis at that parameter's natural scale.
        // tc is perturbed toward the interior of its range.
        let tc_step = if tc0 > 0.5 * (tc_lo + tc_hi) {
            -0.1 * span
        } else {
            0.1 * span
        };
        let mut v1 = init.clone();
        v1[0] += tc_step;
        let mut v2 = init.clone();
        v2[1] += 0.15;
        let mut v3 = init.clone();
        v3[2] += 2.0;

        let solver = NelderMead::new(vec![init, v1, v2, v3]).with_sd_tolerance(1e-8)?;

        let res = match Executor::new(cost.clone(), solver)
            .configure(|state| state.max_iters(MAX_ITERS))
            .run()
        {
            Ok(res) => res,
            Err(_) => continue,
        };

        let run_cost = res.state().get_best_cost();
        if !run_cost.is_finite() {
            continue;
        }
        if let Some(params) = res.state().get_best_param() {
            if best.as_ref().map_or(true, |(c, _)| run_cost < *c) {
                best = Some((run_cost, params.clone()));
            }
        }
        if let Some((c, _)) = &best {
            if *c <= early_exit_sse {
                break;
            }
        }
    }

    let (best_cost, best_params) = match best {
        Some(b) if b.0 < PENALTY => b,
        _ => return Err("fit_argmin: no valid fit found".into()),
    };

    let tc_s = best_params[0];
    let m = best_params[1];
    let w = best_params[2];

    // Recompute the linear parameters exactly at the optimum.
    let (a, b, c1, c2) = matrix_equation(&shifted, price, tc_s, m, w);
    if !(a.is_finite() && b.is_finite() && c1.is_finite() && c2.is_finite()) {
        return Err("fit_argmin: linear solve failed at optimum".into());
    }

    Ok((tc_s + t0, m, w, a, b, c1, c2, best_cost))
}
