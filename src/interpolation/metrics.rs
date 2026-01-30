use std::{fmt::Debug, iter::Sum};

use nalgebra::Point2;
use num::Float;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator, IndexedParallelIterator};

/// Returns the MSE (Mean Square Error) between two sets of points.
/// Points are compared by their y-values at matching indices.
pub fn mse<T>(lhs_points: &[Point2<T>], rhs_points: &[Point2<T>]) -> T
where
    T: Float + Debug + Send + Sync + Sum + 'static,
{
    assert_eq!(
        lhs_points.len(),
        rhs_points.len(),
        "The two arrays of points must have the same length."
    );

    let n = T::from(lhs_points.len()).unwrap();

    lhs_points
        .par_iter()
        .zip(rhs_points.par_iter())
        .map(|(lhs, rhs)| {
            let diff = lhs.y - rhs.y;
            diff * diff
        })
        .sum::<T>()
        / n
}

/// Returns the symmetric Hausdorff distance between two sets of points.
/// Hausdorff distance (https://en.wikipedia.org/wiki/Hausdorff_distance) is a measure of how far two
/// metric spaces are from each other. It is the maximum distance between any point in one space and
/// the closest point in the other space.
pub fn hausdorff<T>(lhs_points: &[Point2<T>], rhs_points: &[Point2<T>]) -> T
where
    T: Float + Debug + Send + Sync + 'static,
{
    // Directed Hausdorff: max over points in A of min distance to B
    let directed_hausdorff = |a: &[Point2<T>], b: &[Point2<T>]| -> T {
        a.par_iter()
            .map(|pa| {
                b.iter()
                    .map(|pb| {
                        // Calculate the distance squared between the points.
                        let dx = pa.x - pb.x;
                        let dy = pa.y - pb.y;
                        dx * dx + dy * dy
                    })
                    // Find the minimum distance between the point and the curve.
                    .fold(T::infinity(), |acc, d| if d < acc { d } else { acc })
                    // Get the square root.
                    .sqrt()
            })
            // Find the maximum distance between the point and the curve.
            .reduce(|| T::zero(), |acc, d| if d > acc { d } else { acc })
    };

    // Use symmetric Hausdorff distance.
    let d_ab = directed_hausdorff(lhs_points, rhs_points);
    let d_ba = directed_hausdorff(rhs_points, lhs_points);
    if d_ab > d_ba { d_ab } else { d_ba }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    fn pt(x: f64, y: f64) -> Point2<f64> {
        Point2::new(x, y)
    }

    /// MSE Tests

    #[test]
    fn mse_identical_arrays() {
        // MSE between identical arrays should be 0
        let points = vec![pt(0.0, 1.0), pt(1.0, 2.0), pt(2.0, 3.0)];
        assert_eq!(mse(&points, &points), 0.0);
    }

    #[test]
    fn mse_known_values() {
        // Manual calculation: differences are [1, 2, 3], squared = [1, 4, 9], mean = 14/3
        let lhs = vec![pt(0.0, 1.0), pt(1.0, 2.0), pt(2.0, 3.0)];
        let rhs = vec![pt(0.0, 0.0), pt(1.0, 0.0), pt(2.0, 0.0)];
        let expected = (1.0 + 4.0 + 9.0) / 3.0;
        assert_approx_eq!(mse(&lhs, &rhs), expected);
    }

    #[test]
    fn mse_symmetry() {
        // MSE should be symmetric since (a-b)^2 = (b-a)^2
        let lhs = vec![pt(0.0, 1.0), pt(1.0, 5.0), pt(2.0, 3.0)];
        let rhs = vec![pt(0.0, 2.0), pt(1.0, 1.0), pt(2.0, 7.0)];
        assert_eq!(mse(&lhs, &rhs), mse(&rhs, &lhs));
    }

    #[test]
    fn mse_single_point() {
        let lhs = vec![pt(0.0, 5.0)];
        let rhs = vec![pt(0.0, 3.0)];
        assert_eq!(mse(&lhs, &rhs), 4.0); // (5-3)^2 = 4
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn mse_different_lengths_panics() {
        let lhs = vec![pt(0.0, 1.0), pt(1.0, 2.0)];
        let rhs = vec![pt(0.0, 1.0)];
        mse(&lhs, &rhs);
    }
    
    /// Hausdorff Tests

    #[test]
    fn hausdorff_identical_arrays() {
        // Hausdorff distance between identical arrays should be 0
        let points = vec![pt(0.0, 0.0), pt(1.0, 1.0), pt(2.0, 2.0)];
        assert_eq!(hausdorff(&points, &points), 0.0);
    }

    #[test]
    fn hausdorff_symmetry() {
        // Symmetric Hausdorff distance should be symmetric by definition
        let a = vec![pt(0.0, 0.0), pt(1.0, 0.0)];
        let b = vec![pt(0.0, 1.0), pt(1.0, 1.0), pt(2.0, 1.0)];
        assert_eq!(hausdorff(&a, &b), hausdorff(&b, &a));
    }

    #[test]
    fn hausdorff_simple_horizontal_shift() {
        // Two points on x-axis, shifted by 1 unit
        let a = vec![pt(0.0, 0.0)];
        let b = vec![pt(1.0, 0.0)];
        assert_approx_eq!(hausdorff(&a, &b), 1.0);
    }

    #[test]
    fn hausdorff_simple_vertical_shift() {
        // Two points on y-axis, shifted by 2 units
        let a = vec![pt(0.0, 0.0)];
        let b = vec![pt(0.0, 2.0)];
        assert_approx_eq!(hausdorff(&a, &b), 2.0);
    }

    #[test]
    fn hausdorff_diagonal_distance() {
        // Distance should be sqrt(1^2 + 1^2) = sqrt(2)
        let a = vec![pt(0.0, 0.0)];
        let b = vec![pt(1.0, 1.0)];
        assert_approx_eq!(hausdorff(&a, &b), 2.0_f64.sqrt());
    }

    #[test]
    fn hausdorff_asymmetric_case() {
        // Test case from SciPy: directed Hausdorff is not symmetric
        // A is a unit circle, B is a circle with radius 2
        let a = vec![pt(1.0, 0.0), pt(0.0, 1.0), pt(-1.0, 0.0), pt(0.0, -1.0)];
        let b = vec![pt(2.0, 0.0), pt(0.0, 2.0), pt(-2.0, 0.0), pt(0.0, -4.0)];
        
        // Symmetric Hausdorff should be the max of both directions
        // From A to B: max distance is from (0, -1) to closest in B
        // From B to A: max distance is from (0, -4) to closest in A = 3.0
        let result = hausdorff(&a, &b);
        assert_approx_eq!(result, 3.0);
    }

    #[test]
    fn hausdorff_known_2d_case() {
        // Simple case with known geometry
        // Triangle vs shifted triangle
        let a = vec![pt(0.0, 0.0), pt(1.0, 0.0), pt(0.5, 1.0)];
        let b = vec![pt(0.0, 2.0), pt(1.0, 2.0), pt(0.5, 3.0)];
        // All points in A are 2 units below corresponding points in B
        assert_approx_eq!(hausdorff(&a, &b), 2.0);
    }
}
