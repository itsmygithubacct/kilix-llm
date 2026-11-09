# Vendoring llama.cpp

Reference for [kilix-llm](../README.md).

The source is a git submodule at a pinned commit, recorded in
`third_party/llama.cpp.pin`. Clone with `--recurse-submodules`, or:

```sh
git submodule update --init third_party/llama.cpp
```

To move the pin:

```sh
./scripts/vendor-sync.sh <40-hex-sha>
```

To change llama.cpp itself, edit files under `third_party/llama.cpp/`. For a
change that should survive the next sync, drop a git patch in `patches/` and
list it, with sha256, in the pin file. `scripts/build-runtime.sh` copies the
vendor tree, applies those patches, and compiles into `build/llama.cpp`
(override `KILIX_LLM_BUILD_DIR`).

CMake uses Release, native CPU, CUDA/Metal off, curl off, and Vulkan on only
when `--backend vulkan` is selected.
Model download is `kilix-llm pull`, not the runtime.
