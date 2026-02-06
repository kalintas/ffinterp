use num::Float;
use std::f64::consts::E;
use std::f64::consts::PI;

pub fn weierstrass<F: Float>(x: F) -> F {
    let a = F::from(0.5).unwrap();
    let b = F::from(3.0).unwrap();
    let pi = F::from(PI).unwrap();

    (0..30).fold(F::zero(), |sum, n| {
        let n_i32 = n as i32;
        sum + a.powi(n_i32) * (b.powi(n_i32) * pi * x).cos()
    })
}

pub fn weierstrass_integral(x: f64) -> f64 {
    let a = 0.5f64;
    let b = 3.0f64;
    let pi = std::f64::consts::PI;

    (0..30)
        .map(|n| {
            let numerator = a.powi(n);
            let denominator = b.powi(n) * pi;
            (numerator / denominator) * (b.powi(n) * pi * x).sin()
        })
        .sum()
}

pub fn wen<F: Float>(x: F) -> F {
    let mut product = F::one();
    let half = F::from(0.5).unwrap();
    let six = F::from(6.0).unwrap();
    let pi = F::from(PI).unwrap();

    for n in 1..=1000 {
        let magnitude = half.powi(n);

        if F::one() + magnitude == F::one() {
            break;
        }

        let angle = six.powi(n) * pi * x;
        let term = F::one() + magnitude * angle.sin();

        product = product * term;
    }

    product
}

pub fn parabol<F: Float>(x: F) -> F {
    let one = F::one();
    let two = F::from(2.0).unwrap();
    one - (two * x - one).powi(2)
}

pub fn riemann<F: Float>(x: F) -> F {
    (1..=500).fold(F::zero(), |sum, k| {
        let k_f = F::from(k).unwrap();
        sum + (F::one() / k_f.powi(2)) * (k_f.powi(2) * x).sin()
    })
}

pub fn ackley<F: Float>(x: F, y: F) -> F {
    let a = F::from(20.0).unwrap();
    let b = F::from(0.2).unwrap();
    let c = F::from(2.0 * PI).unwrap();
    let e = F::from(E).unwrap();
    let half = F::from(0.5).unwrap();

    let term1 = -a * (-b * (half * (x.powi(2) + y.powi(2))).sqrt()).exp();
    let term2 = -(half * ((c * x).cos() + (c * y).cos())).exp();

    term1 + term2 + a + e
}

pub fn rastrigin<F: Float>(x: F, y: F) -> F {
    let twenty = F::from(20.0).unwrap();
    let ten = F::from(10.0).unwrap();
    let two_pi = F::from(2.0 * PI).unwrap();

    twenty + (x.powi(2) - ten * (two_pi * x).cos()) + (y.powi(2) - ten * (two_pi * y).cos())
}

pub fn eggholder<F: Float>(x: F, y: F) -> F {
    let forty_seven = F::from(47.0).unwrap();
    let half = F::from(0.5).unwrap();

    let term1 = -(y + forty_seven) * ((y + x * half + forty_seven).abs().sqrt()).sin();
    let term2 = -x * ((x - (y + forty_seven)).abs().sqrt()).sin();

    term1 + term2
}
