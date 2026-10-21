# Changelog

## 1.1 (2026-10)

Based on llama.cpp v0.7.0, like 1.0.

- Mixture-of-experts models now run their expert weights in Q8A64. Prompt processing is 1.8 to 2.1 times faster than llama.cpp v0.7.0 with Q8_0 on granite-3.1-1b-a400m and OLMoE-1B-7B on the Arm test machine, and 1.5 times faster on the Ryzen 9 5950X, with less divergence from the original weights.
- On Arm CPUs with i8mm, the `linux-arm64-multi` archive runs Q8A64 prompt processing on the i8mm matrix instructions. The output is bit-identical to the dotprod kernels. It has been tested under emulation but not timed on real hardware.
- Q8A64 files of mixture-of-experts models use a new file revision, which VelociQ8 1.0 refuses with an error. Files of dense models are unchanged, and 1.1 loads every file 1.0 loads.
- The archives include `llama-quantize`, which writes Q8A64 files from BF16, F16 or F32 GGUF models.
- Newly checked models: granite-3.1-1b-a400m, OLMoE-1B-7B, LFM2.5-1.2B, Qwen3.5-0.8B and gemma-4-E2B.
- Dense models give the same output as in 1.0, checked bit for bit on the control models.

## 1.0 (2026-10)

First public release, based on llama.cpp v0.7.0.

- Q8A64 weight format with CPU kernels for Arm (dotprod) and x86-64 (AVX2 + FMA), plus a portable C path for other Arm CPUs.
- BF16, F16 and F32 models convert to Q8A64 while loading. Pre-converted Q8A64 files load directly.
- Faster attention for embedding models on Arm.
- Memory locking (`--load-mode mlock`) works with Q8A64 weights on Linux and Windows.
- Release archives for Linux on Arm (single target and multi-CPU), Linux on x86-64 and Windows on x86-64.
