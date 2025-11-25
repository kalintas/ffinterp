mod plotter;

use std::f64::consts::PI;

use ffinterp::interpolation::{one_d::Interpolant1D, FreeVariables, Interpolant};
use nalgebra::Point2;
use plotter::{show_plot, PlotData};

fn wen(x: f64) -> f64 {
    let mut product = 1.0;
    
    for n in 1..=1000 {
        let magnitude = 0.5f64.powi(n);
        
        if 1.0 + magnitude == 1.0 {
            break;
        }

        let angle = 6.0f64.powi(n) * PI * x;
        let term = 1.0 + magnitude * angle.sin();
        
        product *= term;
    }
    
    product
}

fn main() -> Result<(), eframe::Error> {
    let n = 1000;
    let mut points = Vec::with_capacity(n);
    
    for i in 0..n {
        let x = (i as f64 / n as f64) * 2.0 * PI - PI;
        points.push(Point2::new(x, wen(x)));
    }

    let interpolant = Interpolant1D::new(&points, FreeVariables::Scalar(0.01), 10);

    let test_point_count = n * 10;
    let mut test_x = Vec::with_capacity(test_point_count);
    
    for i in 0..test_point_count {
        test_x.push((i as f64 / test_point_count as f64) * 2.0 * PI - PI);
    }
    
    let result = interpolant.evaluate_many(&test_x);

    let data = PlotData {
        real: test_x.iter().map(|&x| [x, wen(x)]).collect(),
        interp: test_x.iter().zip(result.iter()).map(|(&x, &y)| [x, y]).collect(),
        points: points.iter().map(|p| [p.x, p.y]).collect(),
    };

    show_plot(data)
}
