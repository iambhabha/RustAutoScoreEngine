# --- Build Stage ---
FROM rust:1.80-slim as builder

WORKDIR /usr/src/app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy source code and config
COPY . .

# Build for release
RUN cargo build --release

# --- Run Stage ---
FROM debian:bookworm-slim

WORKDIR /usr/src/app

# Install runtime dependencies (Mesa/Vulkan for wgpu fallback)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libvulkan1 \
    mesa-vulkan-drivers \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary from builder
COPY --from=builder /usr/src/app/target/release/rust_auto_score_engine /usr/local/bin/dartvision

# Copy assets
COPY --from=builder /usr/src/app/static ./static
COPY --from=builder /usr/src/app/model_weights.bin ./model_weights.bin

# Environment variables for HF Spaces
ENV PORT=7860
EXPOSE 7860

# Run the GUI
CMD ["dartvision", "gui"]
