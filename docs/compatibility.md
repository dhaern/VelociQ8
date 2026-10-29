# Compatibility

## CPUs and operating systems

| Archive | CPU | Operating system |
|---|---|---|
| `linux-arm64` | 64-bit Arm with Armv8.2-A, dotprod and the crypto extension: Ampere Altra, AWS Graviton2 and later, Arm Neoverse N1 and later | Linux with glibc 2.28 or newer (for example RHEL, AlmaLinux or Rocky 8, Debian 10, Ubuntu 20.04, Amazon Linux 2023 and later). Also needs the GCC runtime `libgcc_s.so.1` and the OpenMP runtime `libgomp.so.1`; if `libgomp.so.1` is missing, install `libgomp1` (Debian, Ubuntu) or `libgomp` (RHEL family) |
| `linux-arm64-multi` | Any 64-bit Arm CPU. At startup it loads the build that best matches the CPU, from Armv8.0 to Armv9.2. Since 1.1, the builds for CPUs with i8mm use it for Q8A64 | Same as `linux-arm64` |
| `linux-x64` | x86-64 with AVX2 and FMA (x86-64-v3): Intel Haswell or newer, AMD Zen or newer | Linux with glibc 2.28 or newer. No other libraries needed |
| `windows-x64` | Same as `linux-x64` | Windows, 64-bit. Tested on Windows 11. The executables are static and need no DLLs |

On Arm CPUs without dotprod, the `multi` archive still runs Q8A64 through portable C code, which is much slower than the dotprod kernels. On x86-64 CPUs without AVX2 and FMA, the x86 archives do not run at all.

The release builds run on the CPU, except the portable Ollama archives for x86-64, which also use the GPU through Vulkan or, in the `-cuda` archives, CUDA (see [usage.md](usage.md#gpu)). The builds have no HTTPS support (see [usage.md](usage.md#https)).

## Models

Q8A64 applies to the matrix weights of any model the underlying llama.cpp release can load, as long as the GGUF stores them in BF16, F16 or F32. These models have been checked for accuracy and speed:

| Model | Type |
|---|---|
| EmbeddingGemma 300M | Embeddings |
| bge-base-en-v1.5 | Embeddings (BERT) |
| Qwen3-1.7B | LLM, also used for embeddings |
| Llama-3.2-1B-Instruct | LLM |
| gemma-3-1b-it | LLM |
| granite-4.0-h-350m | LLM (hybrid with recurrent layers) |
| granite-3.1-1b-a400m-instruct | LLM, mixture of experts (1.1) |
| OLMoE-1B-7B-0125-Instruct | LLM, mixture of experts (1.1) |
| LFM2.5-1.2B-Instruct | LLM (hybrid with convolution layers), accuracy only |
| Qwen3.5-0.8B | LLM, accuracy only |
| gemma-4-E2B-it | LLM, accuracy only |
| LFM3-1.7B, LFM3-1.7B-Base | LLM (hybrid with convolution layers) (1.5) |
| LFM3-VL-2B | LLM with image input (1.5) |

Weights that Q8A64 does not cover keep their original type and run on the normal llama.cpp kernels. The model still works; that part is just not faster.

## Known limitations

- **Mixture-of-experts models in 1.0.** VelociQ8 1.0 keeps the expert weights in their original type, and in an MoE model they are most of the weights, so MoE models run but gain little. VelociQ8 1.1 converts them to Q8A64.
- **Q8A64 files of MoE models need 1.1.** They use a newer file revision, and VelociQ8 1.0 refuses them with `Unsupported file revision or container`. Files of dense models are the same in 1.0 and 1.1.
- **LLMs on Arm.** On the Arm machine, llama.cpp v0.7.0 with KleidiAI ran Llama-3.2-1B 16 % faster than VelociQ8 on prompt processing and 12 % faster on text generation, with 10 to 18 times more divergence from F32, measured on llama.cpp v0.6 (see [benchmarks.md](benchmarks.md#llm) and [quality.md](quality.md#llms)). On the Ryzen 9 5950X, text generation is a tie with llama.cpp Q8_0.
- **Arm CPUs with i8mm** (Graviton3 and later, Neoverse N2/V1/V2, Axion, Cobalt 100). Since 1.1, the `linux-arm64-multi` archive runs Q8A64 prompt processing on the i8mm matrix instructions; text generation stays on the dotprod kernels. The i8mm kernels give bit-identical results to the dotprod kernels. They were tested under emulation of Neoverse N2 and V1, but VelociQ8 has not been timed on any of these CPUs, so there are no speed numbers for them. On Ampere CPUs the archive keeps the dotprod kernels, and the single-target `linux-arm64` archive uses dotprod on every CPU. The rest of the model can run on different llama.cpp code paths on these CPUs, so the full model output is not bit-identical to the A1's. A Q8_0 GGUF is not converted to Q8A64, so it runs on the normal llama.cpp kernels.
- **File size.** When a model uses the token embedding table only for lookups, the Q8A64 file keeps that table in its original precision. This is the case for embedding models and for LLMs with a separate output layer, and their files are bigger than Q8_0: 490 MiB against 313 MiB for EmbeddingGemma 300M. Models that share the table with the output layer store it once, in Q8A64 (Llama-3.2-1B: 1223 MiB against 1260 MiB for Q8_0). MoE files keep the shared table in its original precision as well (granite-3.1-1b-a400m: 1413 MiB against 1356 MiB for Q8_0).
- **Memory while loading.** Converting a BF16 model at load time needs the BF16 weights in memory before the conversion. Pre-converted Q8A64 files avoid that peak; since 1.1 you can write them with `llama-quantize`. On large models the difference matters: running OLMoE-1B-7B from F16 peaked at up to 19.5 GiB of resident memory, against 6.9 GiB from the Q8A64 file. Under a memory limit of 8.2 GiB with no swap, it still ran from F16, but the same short run took 301 s instead of 78 s.
- **No converter in 1.0.** The 1.0 archives do not include `llama-quantize`: they run any BF16, F16 or F32 GGUF directly but cannot write Q8A64 files. The 1.1 archives include it.
- **llama.cpp version.** Each VelociQ8 release is tied to one llama.cpp release (v0.7.0 for VelociQ8 1.0 and 1.1, b11738 for 1.2, b11803 for 1.3, b11845 for 1.4, b11963 for 1.5). Features added to llama.cpp later arrive with the next VelociQ8 release.
