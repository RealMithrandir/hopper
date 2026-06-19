//! Compile `proto/hopper.proto` into Rust with prost. We parse the `.proto` with
//! `protox` (pure Rust) so no system `protoc` is required — CI stays hermetic.

fn main() {
    let descriptors =
        protox::compile(["proto/hopper.proto"], ["proto"]).expect("compile hopper.proto");
    prost_build::Config::new()
        .compile_fds(descriptors)
        .expect("prost codegen from descriptors");
    println!("cargo:rerun-if-changed=proto/hopper.proto");
}
