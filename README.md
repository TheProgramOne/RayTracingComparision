# RayTracingComparision

A performance comparison of the same raytracer implemented natively in four languages: **C**, **Rust**, **Python**, and **JavaScript (Node.js)**.

Each version renders an identical scene — a ground plane and a field of randomly-placed spheres with Lambertian, metal, and dielectric (glass) materials — using a standard Whitted-style raytracer. The goal isn't to build the fastest raytracer possible, but to compare raw language/runtime performance on an identical algorithm.

## Fairness rules

Every port follows the same constraints so the comparison measures the language, not library tricks:

1. **No JSON parsing during the timed render.** The scene (sphere positions, radii, materials, camera) is frozen in `scene_spec.json` and hardcoded as native constants in each language (`scene_data.h`, `scene_data.js`, `scene_data.rs`).
2. **Timing is split into three phases** — scene setup, render, file write — and only the render phase counts toward the benchmark.
3. **No multithreading, no SIMD, no external raytracing libraries.** Every version uses plain single-threaded scalar loops.

## Repo layout

| File | Language | Notes |
|---|---|---|
| `raytracer.c` | C | includes `scene_data.h` |
| `raytracer.py` | Python | reads scene data inline |
| `raytracer.js` | Node.js | requires `scene_data.js` |
| `src/main.rs` + `src/scene_data.rs` | Rust | built via Cargo (see `Cargo.toml`) |
| `scene_spec.json` | — | source-of-truth scene description used to generate the per-language scene data files |

## Building and running

**C**
```bash
gcc -O3 raytracer.c -o raytracer_c -lm
./raytracer_c --width 400 --height 225 --samples 20 --depth 20 --out render.ppm
```

**Rust**
```bash
cargo build --release
./target/release/raytracer --width 400 --height 225 --samples 20 --depth 20 --out render.ppm
```

**Python**
```bash
python3 raytracer.py --width 400 --height 225 --samples 20 --depth 20 --out render.ppm
```

**JavaScript (Node.js)**
```bash
node raytracer.js --width 400 --height 225 --samples 20 --depth 20 --out render.ppm
```

All four accept the same flags (`--width`, `--height`, `--samples`, `--depth`, `--out`) and print timing breakdowns to stdout:

```
scene_setup_seconds: 0.0012
render_seconds:      4.8831
file_write_seconds:  0.0451
total_seconds:       4.9294
```

Output is a `.ppm` image, viewable with most image tools (or convertible with ImageMagick: `convert render.ppm render.png`).

## Benchmarking

For a fair comparison, run all four with identical flags and compare `render_seconds`. Higher `--samples` and `--depth` values increase render time and reduce noise in the timing measurement.
