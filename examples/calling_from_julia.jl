# A basic example that uses FFInterp to 1D interpolate sine wave  
using Plots
using FFInterp

n = 100  
xs = range(-π, π; length=n)
ys = sin.(xs)

# points should be either a Vector{Float} or an array of SVectors
points = collect(Iterators.flatten(zip(xs, ys)))

# This defines the method and the free variable.
# Should either choose a scalar for the whole maps
# Or an array of float with the size of n - 1 
method = FFInterp.Interp1D(0.01)
interpolant = FFInterp.interpolate(points, method, 0.0, 10)

test_point_count = n * 10
test_x = range(-π, π; length=test_point_count)
# Interpolate on the whole test_x array.
interp_y = interpolant(collect(test_x))
# Interpolant can evaluate single points too
# value = interpolant(first(test_x))
# And of course can be used with the broadcast operator
# But this will call Rust for every individual point  
# Therefore its not parallel and has more overhead.
# interp_y = interpolant.(collect(test_x))

p = plot(test_x, sin.(test_x),
    label="Real", linewidth=2, color=:blue)

plot!(p, test_x, interp_y,
    label="Interpolated", linewidth=2, color=:red)

scatter!(p, xs, ys,
    label="Input", markersize=1, color=:black)

display(p)
