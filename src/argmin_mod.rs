use argmin::{
    core::{CostFunction, Error, Executor, State},
    solver::neldermead::NelderMead,
};
use ndarray::{array, Array1};
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use std::error::Error as StdError;
use nalgebra::{DMatrix, DVector as nalgebraDVector};

#[derive(Debug, Clone)]
struct LpplsCostFunction {
    time: Vec<f64>,
    price: Vec<f64>,
}

impl CostFunction for LpplsCostFunction {
    type Param = Array1<f64>;  // Use Array1<f64> for parameters
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        // Use Array1<f64> directly in the cost calculation
        Ok(lppls_cost(
            &param.to_vec(),  // Convert Array1<f64> to Vec<f64>
            &self.time,       // Time data
            &self.price,      // Price data
        ))
    }
}

// The matrix equation for solving parameters
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
        let dt = (tc - time[i]).abs();
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

    let matrix_1 = DMatrix::from_row_slice(4, 4, &[
        n as f64, sum_fi, sum_gi, sum_hi,
        sum_fi, fi_pow_2, figi, fihi,
        sum_gi, figi, gi_pow_2, gihi,
        sum_hi, fihi, gihi, hi_pow_2
    ]);

    let matrix_2 = nalgebraDVector::from_row_slice(&[
        sum_yi, yifi, yigi, yihi
    ]);

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

// LPPLS cost function
pub fn lppls_cost(
    param: &[f64],  // Parameters: tc, m, w, a, b, c1, c2
    time: &[f64],
    price: &[f64],
) -> f64 {
    let tc = param[0];
    let m = param[1];
    let w = param[2];
    let a = param[3];
    let b = param[4];
    let c1 = param[5];
    let c2 = param[6];

    if tc <= time.iter().cloned().fold(f64::INFINITY, f64::min) || m <= 0.0 || w <= 0.0 {
        return f64::INFINITY;
    }

    // Calculate residuals
    time.iter()
        .zip(price.iter())
        .map(|(&t, &p_obs)| {
            let dt = tc - t;
            if dt <= 0.0 {
                return f64::INFINITY;
            }

            let lppls = a
                + b * dt.powf(m)
                + (c1 * dt.powf(m) * (w * dt.ln()).cos())
                + (c2 * dt.powf(m) * (w * dt.ln()).sin());

            (p_obs - lppls).powi(2)
        })
        .sum()
}

// Fit using Argmin and Nelder-Mead
pub fn fit_argmin(time: Vec<f64>, price: Vec<f64>) -> Result<(f64, f64, f64, f64, f64, f64, f64, f64), Box<dyn StdError>> {
    
    // let scaled_price = min_max_scale(&price);
    let cost = LpplsCostFunction { time: time.clone(), price: price.clone() };

    // Generate bounds for the parameters
    let t1 = time.first().unwrap();
    let t2 = time.last().unwrap();
    let tc_lower_bound = (t2 - 60.0).max(t2 - 0.5 * (t2 - t1));
    let tc_upper_bound = (t2 + 252.0).min(t2 + 0.5 * (t2 - t1));
    let m_lower_bound = 0.;
    let m_upper_bound = 1.;
    let w_lower_bound = 2.;
    let w_upper_bound = 15.;

    // Use random uniform distribution for initial parameter values
    let mut rng = thread_rng();
    let tc_between = Uniform::from(tc_lower_bound..=tc_upper_bound);
    let m_between = Uniform::from(m_lower_bound..=m_upper_bound);
    let w_between = Uniform::from(w_lower_bound..=w_upper_bound);

    // Maximum number of retries
    let max_retries = 25;
    let mut retry_count = 0;

    loop {
    
        // Generate initial parameter values
        let tc = tc_between.sample(&mut rng);
        let m = m_between.sample(&mut rng);
        let w = w_between.sample(&mut rng);
    
        // Compute a, b, c1, c2 using matrix equation
        let (a, b, c1, c2) = matrix_equation(&time, &price, tc, m, w);
    
        // Initial parameters for the LPPLS model as Array1<f64>
        let init_param = array![tc, m, w, a, b, c1, c2];
    
        // Set up solver with initial vertices for Nelder-Mead
        let solver = NelderMead::new(vec![
            init_param.clone(),  // First vertex
            init_param.mapv(|x| x + 0.0001),  // Slightly perturbed second vertex
            init_param.mapv(|x| x - 0.0001),  // Slightly perturbed third vertex
        ])
        .with_sd_tolerance(1e-4)?;
    
        // Run the optimization using Executor
        let res = Executor::new(cost.clone(), solver)
            .configure(|state| state.max_iters(100))
            .run()?;

        // Output the best solution
        if let Some(best_params) = res.state().get_param() {
            let best_cost = res.state().get_cost();

            if best_cost > (0.5*(retry_count as f32  + 1.)).into() {
                retry_count += 1;
                if retry_count >= max_retries {
                    println!("Reached maximum retries. Stopping optimization.");
                    break;
                }
            } else {
                // Access elements of Array1<f64> using indexing after unwrapping Option
                let tc = best_params[0];
                let m = best_params[1];
                let w = best_params[2];
                let a = best_params[3];
                let b = best_params[4];
                let c1 = best_params[5];
                let c2 = best_params[6];
                return Ok((tc, m, w, a, b, c1, c2, best_cost));
            }
        } else {
            println!("No best parameters found.");
        }
    }
    Ok((0., 0., 0., 0., 0., 0., 0., 0.))
}
