use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use ffinterp::interpolation::{FreeVariables, Interpolant, one_d::Interpolant1D};
use nalgebra::Point2;

fn setup_interpolant(n_points: usize) -> Interpolant1D<f64> {
    let mut points = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let x = i as f64 / n_points as f64;
        points.push(Point2::new(x, x.sin()));
    }
    Interpolant1D::new(&points, FreeVariables::Scalar(0.01), 10)
}

pub fn bench_evaluation(c: &mut Criterion) {
    let interpolant = setup_interpolant(10000);

    let mut group = c.benchmark_group("1d_sine_wave_interpolation");
    for &n_points in &[10_000, 100_000, 1_000_000] {
        let input_size = n_points;
        let inputs: Vec<f64> = (0..input_size)
            .map(|i| i as f64 / input_size as f64)
            .collect();

        group.bench_function(format!("sequential_{}_pts", input_size), |b| {
            b.iter(|| {
                let res: Vec<f64> = inputs.iter().map(|&x| interpolant.evaluate(x)).collect();
                black_box(res)
            })
        });

        group.bench_function(format!("parallel_{}_pts", input_size), |b| {
            b.iter(|| black_box(interpolant.evaluate_many(black_box(&inputs))))
        });
    }
    group.finish();
}

pub fn bench_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("1d_interpolant_creation");

    for &n_points in &[1000usize, 5000, 10_000, 20_000, 100_000, 1_000_000] {
        group.bench_with_input(format!("create_{}pts", n_points), &n_points, |b, &n| {
            let mut points = Vec::with_capacity(n);
            for i in 0..n {
                let x = i as f64 / n as f64;
                points.push(Point2::new(x, x.sin()));
            }

            b.iter(|| {
                black_box(Interpolant1D::new(
                    black_box(&points),
                    black_box(FreeVariables::Scalar(0.01)),
                    black_box(10),
                ))
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_evaluation, bench_creation);
criterion_main!(benches);
