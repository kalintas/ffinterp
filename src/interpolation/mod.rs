use num::Float;

pub mod one_d;

#[derive(Clone, Debug)]
pub enum FreeVariables<T> {
    Scalar(T),
    Array(Vec<T>),
}

#[derive(Debug, Clone, Copy)]
struct AffineMap<T> {
    a: T,
    c: T,
    d: T,
    e: T,
    f: T,
    end_x: T,
}

pub trait Interpolant {
    type Scalar: Float + Clone;

    fn evaluate(&self, x: Self::Scalar) -> Self::Scalar;
    fn evaluate_many(&self, points: &[Self::Scalar]) -> Vec<Self::Scalar>;
}
