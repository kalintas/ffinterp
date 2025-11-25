use std::{
    cmp::Ordering,
    fmt::Debug,
    ops::{AddAssign, MulAssign},
};

use nalgebra::Point2;
use num::{Float, Zero};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::interpolation::{AffineMap, FreeVariables, Interpolant};

#[derive(Debug, Clone)]
pub struct Interpolant1D<T: Float + Debug + 'static> {
    points: Vec<Point2<T>>,
    maps: Vec<AffineMap<T>>,
    iterations: usize,
}

impl<T> Interpolant1D<T>
where
    T: Debug + Float + 'static + Send + Sync,
{
    pub fn new(points: &[Point2<T>], free_variables: FreeVariables<T>, iterations: usize) -> Self {
        let n = points.len();

        if n <= 1 {
            panic!("More than one point is required to create the Interpolant.");
        }

        let mut points: Vec<Point2<T>> = points.to_vec();
        points.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(Ordering::Equal));

        let p0 = points.first().unwrap();
        let pn = points.last().unwrap();
        let total_x_range = pn.x - p0.x;

        let maps = (0..(n - 1))
            .into_par_iter()
            .map(|i| {
                let p = points[i];
                let p_next = points[i + 1];

                let di = match &free_variables {
                    FreeVariables::Scalar(variable) => *variable,
                    FreeVariables::Array(array) => array[i],
                };

                let a = (p_next.x - p.x) / total_x_range;
                let e = (pn.x * p.x - p0.x * p_next.x) / total_x_range;
                let c = (p_next.y - p.y) / total_x_range - di * (pn.y - p0.y) / total_x_range;
                let f = (pn.x * p.y - p0.x * p_next.y) / total_x_range
                    - di * (pn.x * p0.y - p0.x * pn.y) / total_x_range;

                AffineMap {
                    a,
                    c,
                    d: di,
                    e,
                    f,
                    end_x: p_next.x,
                }
            })
            .collect();

        Self {
            points,
            maps,
            iterations,
        }
    }
}

impl<T> Interpolant for Interpolant1D<T>
where
    T: Float + Debug + AddAssign + MulAssign + Send + Sync,
{
    type Scalar = T;

    fn evaluate(&self, mut x: Self::Scalar) -> Self::Scalar {
        let first_point = self.points.first().unwrap();
        let last_point = self.points.last().unwrap();

        if x <= first_point.x {
            return first_point.y;
        }
        if x >= last_point.x {
            return last_point.y;
        }

        let mut y_accumulated = Zero::zero();
        let mut d_product = T::from(1.0).unwrap();

        for _ in 0..self.iterations {
            let map_index = self.maps.partition_point(|map| map.end_x <= x);
            let map = self.maps[map_index];

            let x_prev = (x - map.e) / map.a;
            let term = map.c * x_prev + map.f;

            y_accumulated += d_product * term;
            d_product *= map.d;
            x = x_prev;

            if d_product.abs() < T::from(1e-9).unwrap() {
                break;
            }
        }
        y_accumulated
    }

    fn evaluate_many(&self, points: &[Self::Scalar]) -> Vec<Self::Scalar> {
        points.par_iter().map(|&x| self.evaluate(x)).collect()
    }
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    use super::*;

    #[test]
    fn interpolant1d_evaluate_works() {
        let n = 1000;
        let mut points = Vec::<Point2<f64>>::with_capacity(n);
        for i in 0..n {
            let x = i as f64 / n as f64;
            points.push(Point2::new(x, x.sin()));
        }

        let interpolant = Interpolant1D::new(&points, FreeVariables::Scalar(0.01), 10);

        points.iter().for_each(|point| {
            let value = interpolant.evaluate(point.x);
            assert_approx_eq!(value, point.y);
        });

        let test_points_n = n * 5;

        for i in 0..test_points_n {
            let x = i as f64 / n as f64;
            let value = interpolant.evaluate(x);
            // TODO: figure out the acceptable error.
            //assert_approx_eq!(value, x.sin());
        }
    }
}
