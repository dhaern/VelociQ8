# Changelog

## 1.3 (2026-10-24)

Based on llama.cpp b11803, the release that Ollama 0.40.9 uses.

- GPU support in the portable Ollama archives for x86-64 (Linux and Windows), through the official Vulkan backend of Ollama 0.40.9, on NVIDIA, AMD and Intel GPUs. Layers that go to the GPU are converted from Q8A64 to Q8_0 while loading, keeping the Q8A64 rotation, and run on the llama.cpp Q8_0 kernels; the rest stays on the CPU with the VelociQ8 kernels. The same Q8A64 file works on both. Tested with a software Vulkan driver; not yet run on a physical GPU.
- New `-cuda` Ollama archives for NVIDIA GPUs (Turing or newer), with the official CUDA 13 backend of Ollama. Only on GitHub, because of their size.
- LoRA adapters load on rotated weights, and a LoRA on the token embeddings of a model with a tied output now works instead of stopping the server.
- Without a GPU, Q8A64 output is the same as in 1.2, checked bit for bit on the control models.

## 1.2 (2026-10-22)

Based on llama.cpp b11738, the release that Ollama 0.40.8 uses.

- Ollama support: portable Ollama 0.40.8 archives for the CPU, for Linux (Arm and x86-64) and Windows, with the official `ollama` executable and the VelociQ8 server. They load Q8A64 files and convert BF16, F16 and F32 models while loading, and do not touch an Ollama already installed. Tested on Linux on Arm; the Windows archive has not been run on Windows yet.
- Q8A64 files of embedding models now carry their pooling type and context length in the file header, which Ollama needs to detect them. Older files still load in every version; the files on Hugging Face have been updated.
- The archives include `llama`, the single command that the Llama app for Windows runs, and `llama-tts` for text-to-speech models.
- Q8A64 output is the same as in 1.1, checked bit for bit on the control models.

## 1.1 (2026-10-21)

Based on llama.cpp v0.7.0, like 1.0.

- Mixture-of-experts models now run their expert weights in Q8A64. Prompt processing is 1.8 to 2.1 times faster than llama.cpp v0.7.0 with Q8_0 on granite-3.1-1b-a400m and OLMoE-1B-7B on the Arm test machine, and 1.5 times faster on the Ryzen 9 5950X, with less divergence from the original weights.
- On Arm CPUs with i8mm, the `linux-arm64-multi` archive runs Q8A64 prompt processing on the i8mm matrix instructions. The output is bit-identical to the dotprod kernels. It has been tested under emulation but not timed on real hardware.
- Q8A64 files of mixture-of-experts models use a new file revision, which VelociQ8 1.0 refuses with an error. Files of dense models are unchanged, and 1.1 loads every file 1.0 loads.
- The archives include `llama-quantize`, which writes Q8A64 files from BF16, F16 or F32 GGUF models.
- Newly checked models: granite-3.1-1b-a400m, OLMoE-1B-7B, LFM2.5-1.2B, Qwen3.5-0.8B and gemma-4-E2B.
- Dense models give the same output as in 1.0, checked bit for bit on the control models.

## 1.0 (2026-10-21)

First public release, based on llama.cpp v0.7.0.

- Q8A64 weight format with CPU kernels for Arm (dotprod) and x86-64 (AVX2 + FMA), plus a portable C path for other Arm CPUs.
- BF16, F16 and F32 models convert to Q8A64 while loading. Pre-converted Q8A64 files load directly.
- Faster attention for embedding models on Arm.
- Memory locking (`--load-mode mlock`) works with Q8A64 weights on Linux and Windows.
- Release archives for Linux on Arm (single target and multi-CPU), Linux on x86-64 and Windows on x86-64.
