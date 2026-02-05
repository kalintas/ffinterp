## FFInterp
A fast fractal interpolation library written in Rust. 

### Building the library 
```
git clone https://github.com/kalintas/ffinterp
cargo build -r
```

### Building with CUDA Enabled

**Prerequisites:**
- NVIDIA GPU with Compute Capability 5.0 (Maxwell) or later
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 12.0+
- **LLVM 7.x** — Set `LLVM_CONFIG` env var or ensure `llvm-config --version` returns `7.x.x`

> ⚠️ LLVM 7.x is old and may require manual installation. Docker images with CUDA and LLVM 7 pre-installed are available in the [Rust CUDA repo](https://github.com/Rust-GPU/rust-cuda/tree/main/container).

```bash
cargo build -r --features cuda
```

See the [Rust CUDA Getting Started Guide](https://rust-gpu.github.io/rust-cuda/guide/getting_started.html) for detailed setup instructions.

### Running examples 
```
cargo run -r --example one_d_sine_wave_interpolation
```

### Running Julia bindings example
First build the project with cargo build -r, then go into the Julia REPL and add the bindings as a development dependency.
```
cargo build -r
julia --project=.
pkg> dev bindings/julia/FFInterp
include("examples/calling_from_julia.jl")
```

### Running Python bindings example
To run the python example, first create a virtual environment and install the requirements. Then use maturin develop to create a development environment for the Python bindings.
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r examples/requirements.txt
maturin develop
python -m examples/calling_from_python
```

### Running Criterion benchmarks
```
cargo bench
```

### Profiling
Profiling in this project is done with [samply](https://github.com/mstange/samply) and [cargo-instruments](https://github.com/cmyr/cargo-instruments). cargo-instruments works only on the macOS. And you will need the Xcode instruments to run it.
Here is how to profile an example.
```
samply record cargo run -r --example one_d_print_sine_wave_integral
```
Or
```
cargo instruments --release --example one_d_print_sine_wave_integral -t "Time Profiler"
```

### TODO
- [ ] Add more documentation 
- [ ] Increase test coverage 
