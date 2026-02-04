mod plotter;

use ffinterp::interpolation::{FreeVariables, Interpolant, one_d::Interpolant1D};
use ffinterp::test_functions;
use nalgebra::Point2;
use plotter::interactive::{show_plot_interactive, PlotData, PlotSettings};

/// Blancmange curve: A self-affine fractal function
fn blancmange(x: f64) -> f64 {
    let s = |val: f64| {
        let frac = val - val.floor();
        if frac < 0.5 { frac } else { 1.0 - frac }
    };
    (0..20).map(|n| 0.5f64.powi(n) * s(2.0f64.powi(n) * x)).sum()
}

/// Multifractal: A function with varying local fractal dimension
fn multifractal(x: f64) -> f64 {
    let d_base = 0.1 + 0.7 * (x.cos() * 0.5 + 0.5);
    (0..15).map(|n| {
        let frequency = 2.0f64.powi(n);
        let amplitude = d_base.powi(n);
        let triangle = ((frequency * x).fract() - 0.5).abs() * 2.0;
        amplitude * triangle
    }).sum()
}

/// Takagi function (also called Blancmange): Classic self-similar fractal
fn takagi(x: f64) -> f64 {
    let s = |val: f64| {
        let frac = (val - val.floor()).abs();
        (frac - 0.5).abs()
    };
    (0..25).map(|n| 0.5f64.powi(n) * s(2.0f64.powi(n) * x)).sum()
}

/// Devil's Staircase (Cantor function): A singular continuous function
fn devils_staircase(x: f64) -> f64 {
    // Approximation using ternary expansion
    let mut result = 0.0;
    let mut power_of_2 = 0.5;
    let mut current = x.clamp(0.0, 1.0);
    
    for _ in 0..50 {
        current *= 3.0;
        let digit = current.floor() as i32;
        current -= digit as f64;
        
        if digit == 0 {
            // Do nothing, continue subdividing
        } else if digit == 1 {
            result += power_of_2;
            return result; // In the middle third, return immediately
        } else {
            result += power_of_2;
        }
        power_of_2 *= 0.5;
    }
    result
}

fn ackley_slice(x: f64) -> f64 {
    test_functions::ackley(x, 0.0)
}

fn rastrigin_slice(x: f64) -> f64 {
    test_functions::rastrigin(x, 0.0)
}

fn get_function(idx: usize) -> fn(f64) -> f64 {
    match idx {
        0 => test_functions::weierstrass,
        1 => blancmange,
        2 => multifractal,
        3 => takagi,
        4 => devils_staircase,
        5 => f64::sin,
        6 => test_functions::wen,
        7 => test_functions::parabol,
        8 => test_functions::riemann,
        9 => ackley_slice,
        10 => rastrigin_slice,
        _ => test_functions::weierstrass,
    }
}

fn get_function_names() -> &'static [&'static str] {
    &[
        "Weierstrass", 
        "Blancmange", 
        "Multifractal", 
        "Takagi", 
        "Devil's Staircase", 
        "Sine Wave", 
        "Wen",
        "Parabol",
        "Riemann",
        "Ackley (y=0)",
        "Rastrigin (y=0)",
        "Eggholder (y=0)"
    ]
}

fn main() {
    let function_names = get_function_names();
    show_plot_interactive(function_names, move |settings: &PlotSettings| {
        let n = settings.n;
        let factor = settings.test_factor;
        let range_start = settings.range_start;
        let range_end = settings.range_end;
        let func = get_function(settings.selected_function);
        
        let points: Vec<Point2<f64>> = (0..n)
            .map(|i| {
                let x = range_start + (i as f64 / (n - 1) as f64) * (range_end - range_start);
                Point2::new(x, func(x))
            })
            .collect();

        let free_vars = if settings.use_individual_d {
            FreeVariables::Array(settings.d_values.clone())
        } else {
            FreeVariables::Scalar(settings.d_scalar)
        };
        
        let interpolant = Interpolant1D::new(&points, free_vars, 100);

        let mut test_x = Vec::new();
        for i in 0..(n - 1) {
            let x_s = points[i].x;
            let x_e = points[i + 1].x;
            for t in 0..factor {
                test_x.push(x_s + (t as f64 / factor as f64) * (x_e - x_s));
            }
        }
        test_x.push(points[n - 1].x);

        let results = interpolant.evaluate_many(&test_x);
        
        let mse: f64 = test_x.iter()
            .zip(results.iter())
            .map(|(&x, &y_interp)| {
                let y_real = func(x);
                (y_real - y_interp).powi(2)
            })
            .sum::<f64>() / test_x.len() as f64;
        
        let integral = interpolant.integrate();

        let test_points: Vec<Point2<f64>> = test_x.iter().map(|&x| Point2::new(x, func(x))).collect();
        let hausdorff = interpolant.get_hausdorff(&test_points);

        let interp_points: Vec<[f64; 2]> = test_x.iter().zip(results.iter()).map(|(&x, &y)| [x, y]).collect();
        let real_points: Vec<[f64; 2]> = test_x.iter().map(|&x| [x, func(x)]).collect();

        PlotData {
            real: real_points,
            interp: interp_points,
            points: points.iter().map(|p| [p.x, p.y]).collect(),
            mse,
            integral,
            hausdorff
        }
    });
}
