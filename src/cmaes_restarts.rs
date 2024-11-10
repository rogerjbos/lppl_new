use cmaes::{CMAESOptions, DVector};
use cmaes::restart::{RestartOptions, RestartStrategy};
use nalgebra::DVector as nalgebraDVector;
use ndarray::Array1;
use std::error::Error as StdError;
use num_format::{Locale, ToFormattedString};
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;

fn format_f64_with_commas(n: f64) -> String {
    let int_part = n.trunc() as i64;
    let frac_part = (n.fract() * 100.0).round() as u64;
    format!("{}.{}", int_part.to_formatted_string(&Locale::en), format!("{:02}", frac_part))
}

struct LPPLS<'a> {
    time: &'a [f64],
    price: &'a [f64],
}

impl<'a> LPPLS<'a> {
    fn cost(&self, p: &Array1<f64>) -> f64 {
        let tc = p[0];
        let m = p[1];
        let w = p[2];
        let residuals: f64 = self.time.iter().zip(self.price.iter()).map(|(&t, &p_obs)| {
            let dt = tc - t;
            if dt <= 0.0 {
                return f64::INFINITY;
            }
            let lppls = p[3] + p[4] * dt.powf(m) + p[5] * dt.powf(m) * (w * dt.ln()).cos() + p[6] * dt.powf(m) * (w * dt.ln()).sin();
            (p_obs - lppls).powi(2)
        }).sum();
        residuals
    }
}

pub fn fit_cmaes(time: Vec<f64>, price: Vec<f64>) -> Result<(), Box<dyn StdError>> {
    let problem = LPPLS { time: &time, price: &price };

    let t1 = time.first().expect("time value");
    let t2 = time.last().expect("time value");
    let tc_lower_bound = (t2 - 60.0).max(t2 - 0.5 * (t2 - t1));
    let tc_upper_bound = (t2 + 252.0).min(t2 + 0.5 * (t2 - t1));
    let m_lower_bound = 0.;
    let m_upper_bound = 1.;
    let w_lower_bound = 2.;
    let w_upper_bound = 15.;

    let mut rng = thread_rng();
    let tc_between = Uniform::from(tc_lower_bound..=tc_upper_bound);
    let m_between = Uniform::from(m_lower_bound..=m_upper_bound);
    let w_between = Uniform::from(w_lower_bound..=w_upper_bound);
    
    let tc = tc_between.sample(&mut rng);
    let m = m_between.sample(&mut rng);
    let w = w_between.sample(&mut rng);

    let (a, b, c1, c2) = (0.0, 0.0, 0.0, 0.0); // Placeholder values; replace with proper initialization

    let init_param = vec![tc, m, w, a, b, c1, c2, t1.clone(), t2.clone()];
    println!("init_params: {:?}", init_param);

    // Define Restart Strategy (Limit restarts to as few as possible)
    let restart_strategy = RestartStrategy::IPOP(Default::default());  // Simpler restart strategy with fewer restarts

    // Define search range for the parameters
    let search_range = -1.0..=1.0;  // Adjust this based on your parameter ranges
    let dim = init_param.len();  // Number of dimensions in the parameter space

    // Create RestartOptions with a simple stopping criterion
    let restarter = RestartOptions::new(dim, search_range, restart_strategy)
        .fun_target(1e-4)  // Set the target function value for convergence
        .enable_printing(true)  // Print progress
        .build()
        .unwrap();

    // Run the optimization with automatic restarts
    let results = restarter.run(|| {
        let problem_ref = &problem;  // Borrow problem immutably
        move |params: &DVector<f64>| {
            let params_array = Array1::from(params.as_slice().to_vec());
            problem_ref.cost(&params_array)  // Use the borrowed reference
        }
    });

    // Retrieve the best solution and cost
    if let Some(best_solution) = results.best {
        let best_cost = best_solution.value;
        let tc = best_solution.point[0];
        let m = best_solution.point[1];
        let w = best_solution.point[2];
        let a = best_solution.point[3];
        let b = best_solution.point[4];
        let c1 = best_solution.point[5];
        let c2 = best_solution.point[6];

        println!(
            "tc: {:.2} m: {:.2} w: {:.2} a: {:.2} b: {:.2} c1: {:.2} c2: {:.2} cost: {}",
            tc, m, w, a, b, c1, c2, format_f64_with_commas(best_cost)
        );
    } else {
        println!("No best solution found");
    }

    Ok(())
}