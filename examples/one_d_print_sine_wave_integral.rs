mod plotter;

use std::{f64::consts::PI, hint::black_box};

use ffinterp::interpolation::{FreeVariables, Interpolant, one_d::Interpolant1D};
use nalgebra::Point2;

fn main() {
    let n = 1000000;
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        let x = (i as f64 / n as f64) * 2.0 * PI - PI;
        points.push(Point2::new(x, x.sin()));
    }

    let interpolant = black_box(Interpolant1D::new(&points, FreeVariables::Scalar(0.01), 10));

    let test_point_count = n * 10;
    let mut test_x = Vec::with_capacity(test_point_count);
    for i in 0..test_point_count {
        test_x.push((i as f64 / test_point_count as f64) * 2.0 * PI - PI);
    }

    let result = interpolant.evaluate_many(&test_x);
    
    println!("{}", interpolant.integrate());
    
    black_box(result);
}
