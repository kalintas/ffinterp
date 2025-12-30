use std::fmt::Debug;

use num::Float;

pub mod one_d;

#[derive(Clone, Debug)]
pub enum FreeVariables<T> {
    Scalar(T),
    Array(Vec<T>),
}

#[derive(Debug, Clone, Copy)]
struct IFSMap<T> {
    a: T,
    d: T,
    e: T,
    q: [T; 4],
    end_x: T,
}

pub trait Interpolant {
    type Scalar: Float + Clone + Debug + 'static;

    fn evaluate(&self, x: Self::Scalar) -> Self::Scalar;
    fn evaluate_many(&self, points: &[Self::Scalar]) -> Vec<Self::Scalar>;
}
