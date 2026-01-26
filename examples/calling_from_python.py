"""
A basic example that uses FFInterp to 1D interpolate a sine wave.
"""
import numpy as np
import matplotlib.pyplot as plt

# Import the ffinterp module (built with maturin)
import ffinterp

n = 100
xs = np.linspace(-np.pi, np.pi, n)
ys = np.sin(xs)

# Points should be a flat numpy array of (x, y) pairs: [x0, y0, x1, y1, ...]
points = np.empty(n * 2, dtype=np.float64)
points[0::2] = xs
points[1::2] = ys

# Create the interpolant with:
# - free_variable: 0.01 (controls fractal behavior, 0 = pure cubic Hermite spline)
# - iterations: 10 (number of IFS iterations)
interpolant = ffinterp.interpolate(points, 0.01, 10)

test_point_count = n * 10
test_x = np.linspace(-np.pi, np.pi, test_point_count)

# Interpolate on the whole test_x array (parallel evaluation in Rust)
interp_y = interpolant.evaluate_many(test_x)

# Interpolant can evaluate single points too
# value = interpolant.evaluate(test_x[0])
# Or using callable syntax:
# value = interpolant(test_x[0])

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(test_x, np.sin(test_x), label="Real", linewidth=2, color="blue")
plt.plot(test_x, interp_y, label="Interpolated", linewidth=2, color="red", linestyle="--")
plt.scatter(xs, ys, label="Input", s=10, color="black", zorder=5)

plt.xlabel("x")
plt.ylabel("y")
plt.title("FFInterp 1D Fractal Interpolation Example")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
