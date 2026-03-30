pub mod args;
pub mod config;
pub mod data;
pub mod inference;
pub mod loss;
pub mod model;
pub mod scoring;
pub mod server;
pub mod tests;
pub mod train;

// WASM Module for the Web Build
#[cfg(target_family = "wasm")]
pub mod wasm_bridge;
