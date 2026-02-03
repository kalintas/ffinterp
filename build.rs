#[cfg(feature = "cuda")]
use cuda_builder::CudaBuilder;

#[cfg(feature = "cuda")]
fn build_cuda() {
    use std::env;
    use std::path;

    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=kernels");

    let out_dir = path::PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = path::PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // Compile the `kernels` crate to `$OUT_DIR/kernels.ptx`.
    CudaBuilder::new(manifest_dir.join("kernels"))
        .copy_to(out_dir.join("kernels.ptx"))
        .build()
        .unwrap();
}

fn main() {
    #[cfg(feature = "cuda")]
    {
        build_cuda();
    }
}
