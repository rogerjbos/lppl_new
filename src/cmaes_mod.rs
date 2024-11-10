use cmaes::{CMAESOptions, DVector}; //, TerminationReason
use ndarray::Array1;
use std::error::Error as StdError;
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;

use lppl_new::matrix_equation;

#[derive(Clone)]
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
                return f64::INFINITY; // Return infinity if the value is invalid
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

    // Maximum number of retries
    let max_retries = 5;
    let mut retry_count = 0;

    loop {
        let problem_clone = problem.clone(); // Clone problem on each iteration

        let tc = tc_between.sample(&mut rng);
        let m = m_between.sample(&mut rng);
        let w = w_between.sample(&mut rng);
        let (a, b, c1, c2) = matrix_equation(&time, &price, tc, m, w);
        let init_param = vec![tc, m, w, a, b, c1, c2];
    
        // Create CMAESOptions and provide the objective function during the build
        let mut optimizer = CMAESOptions::new(DVector::from_vec(init_param.clone()), 0.5)
            .fun_target(1e-4)  // Set the target function value for convergence
            .build(move |params: &DVector<f64>| {
                // Convert params to Array1 for the cost function
                let params_array = Array1::from(params.as_slice().to_vec());
                problem_clone.cost(&params_array)  // Evaluate the cost function
            })
            .unwrap();

        // Run the optimization (no restarts)
        let termination_data = optimizer.run();

        // Check if the best solution exists and if its cost is greater than 100
        if let Some(best_solution) = termination_data.overall_best {
            let best_cost = best_solution.value;

            if best_cost > 200.0 {
                retry_count += 1;
                // Retry up to `max_retries` times
                if retry_count >= max_retries {
                    println!("Reached maximum retries. Stopping optimization.");
                    break;
                }
            } else {
                // Retrieve the best solution and print the values
                let tc = best_solution.point[0];
                let m = best_solution.point[1];
                let w = best_solution.point[2];
                let a = best_solution.point[3];
                let b = best_solution.point[4];
                let c1 = best_solution.point[5];
                let c2 = best_solution.point[6];

                println!(
                    "cmaes: tc: {:.2} m: {:.2} w: {:.2} a: {:.2} b: {:.2} c1: {:.2} c2: {:.2} cost: {:.2}",
                    tc, m, w, a, b, c1, c2, best_cost
                );
                break;
            }
        } else {
            println!("No best solution found.");
            break;
        }
    }

    Ok(())
}
