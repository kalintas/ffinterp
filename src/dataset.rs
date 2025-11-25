use crate::primitives::point::{Point, Point2};
use num::{Float, Zero};
use rgmsh::Gmsh;

pub struct Dataset<T: Float + Zero, const D: usize> {
    points: Vec<Point<T, D>>,
}

impl<T: Float + Zero, const D: usize> Dataset<T, D> {
    pub fn builder<'a>() -> DatasetBuilder<'a, T, D> {
        DatasetBuilder::new()
    }
}

pub struct DatasetBuilder<'a, T: Float + Zero, const D: usize> {
    vertex_points: Option<&'a [Point<T, D>]>,
    lc: Option<T>,
    gridded: bool,
}

pub fn gmsh_model_and_extract_points<T: Float + Zero + Into<f64>, const D: usize>(
    vertex_points: &[Point<T, D>],
    _lc: T, // local size ignored
) -> Vec<Point<T, D>> {
    // Initialize Gmsh
    let gmsh = Gmsh::initialize().expect("Failed to initialize Gmsh");
    let mut model = gmsh
        .create_native_model("t1")
        .expect("Failed to create model");

    // Add vertex points
    let mut point_tags = Vec::new();
    for p in vertex_points.iter() {
        let x = p[0].into();
        let y = if D > 1 { p[1].into() } else { 0.0 };
        let tag = model.add_point(x, y, 0.0).expect("Failed to add point");
        point_tags.push(tag);
    }

    // Build geometry
    match vertex_points.len() {
        2 => {
            // Line
            model
                .add_line(point_tags[0], point_tags[1])
                .expect("Failed to add line");
        }
        n if n > 2 => {
            // Polygon
            let mut curve_tags = Vec::new();
            for i in 0..n {
                let start = point_tags[i];
                let end = point_tags[(i + 1) % n];
                let curve = model.add_line(start, end).expect("Failed to add line");
                curve_tags.push(curve);
            }
            let wire = model
                .add_curve_loop(&curve_tags)
                .expect("Failed to add curve loop");
            model
                .add_plane_surface(wire)
                .expect("Failed to add plane surface");
        }
        _ => {}
    }

    // Generate mesh
    let dim = if D > 1 { 2 } else { 1 };
    model.generate_mesh(dim).unwrap();

    extract_node_coordinates().unwrap()
}

fn extract_node_coordinates<T, const D: usize>()
-> Result<Vec<Point<T, D>>, Box<dyn std::error::Error>> {
    // Ensure GMSH is initialized
    // Retrieve node tags and coordinates
    let mut node_tags: Vec<i32> = Vec::new();
    let mut coords: Vec<f64> = Vec::new();
    let mut num_tags = 0;
    let mut num_coords = 0;

    // Check if the retrieval was successful
    if num_coords > 0 {
        // Process the node coordinates
        for i in 0..(num_coords / 3) {
            let x = coords[3 * i];
            let y = coords[3 * i + 1];
            let z = coords[3 * i + 2];
            println!("Node {}: ({}, {}, {})", node_tags[i], x, y, z);
        }
    } else {
        println!("No nodes found.");
    }

    Ok(Vec::new())
}

impl<'a, T: Float + Zero, const D: usize> DatasetBuilder<'a, T, D> {
    pub fn new() -> Self {
        Self {
            vertex_points: None,
            lc: None,
            gridded: false,
        }
    }

    pub fn vertex_points(mut self, vertex_points: &'a [Point<T, D>]) -> Self {
        self.vertex_points = Some(vertex_points);
        self
    }
    pub fn lc(mut self, lc: T) -> Self {
        self.lc = Some(lc);
        self
    }
    pub fn gridded(mut self, gridded: bool) -> Self {
        self.gridded = gridded;
        self
    }

    pub fn build(self) -> Dataset<T, D> {
        assert_ne!(D, 0, "Dimension should be greater than 0.");

        if D == 1 {
            if self.gridded {
            } else {
            }
        } else {
            // D > 1
            if self.gridded {
            } else {
            }
        }

        Dataset { points: Vec::new() }
    }
}

pub fn vertex_points<T: Float + Zero>(point_count: usize, p0: Point2<T>, r: T) -> Vec<Point2<T>> {
    let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap(); // convert 2π to generic float
    (0..point_count)
        .map(|i| {
            let theta = T::from(i).unwrap() / T::from(point_count).unwrap() * two_pi;
            let x = r * theta.cos();
            let y = r * theta.sin();
            Point2::<T>::new([p0.x() + x, p0.y() + y])
        })
        .collect()
}

pub fn create_dataset_transformed<T, F, const D: usize, const DO: usize>(
    func: F,
    vertex_points: Vec<Point<T, D>>,
) -> Vec<Point<T, DO>>
where
    T: Float + Zero,
    F: Fn(Point<T, D>) -> Point<T, DO>,
{
    let dataset = create_dataset(vertex_points);
    dataset.into_iter().map(func).collect()
}

pub fn create_dataset<T: Float, const D: usize>(
    vertex_points: Vec<Point<T, D>>,
) -> Vec<Point<T, D>> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::point;
    use approx::assert_relative_eq; // for floating-point comparison

    #[test]
    fn test_vertex_points_square() {
        let center = Point2::new([0.0, 0.0]);
        let radius = 1.0;
        let n = 4;
        let points = vertex_points(n, center, radius);

        // There should be exactly n points
        assert_eq!(points.len(), n);

        for (i, p) in points.iter().enumerate() {
            // Distance from center should be approximately equal to radius
            let dx = p.x() - center.x();
            let dy = p.y() - center.y();
            let distance = (dx * dx + dy * dy).sqrt();
            assert_relative_eq!(distance, radius, epsilon = f64::EPSILON);

            // Check angles roughly (optional, for more precise testing)
            let expected_angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            let angle = dy.atan2(dx);
            // atan2 returns [-π, π], adjust to [0, 2π]
            let angle = if angle < 0.0 {
                angle + 2.0 * std::f64::consts::PI
            } else {
                angle
            };
            assert_relative_eq!(angle, expected_angle, epsilon = f64::EPSILON);
        }
    }

    #[test]
    fn test_dataset_builder() {
        DatasetBuilder::new()
            .gridded(false)
            .vertex_points(&[point!(0.0), point!(1.0)])
            .build();
    }
}
