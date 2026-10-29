# Quality

Every 8-bit format loses some accuracy. The reference here is always the same model in full precision, never another 8-bit format: F32, or the model's original BF16 or F16 weights where a table says so.

## Embeddings

These are the same runs as the Arm table in [benchmarks.md](benchmarks.md): EmbeddingGemma 300M, 300 real documents and 300 real searches, against the F32 model run by llama.cpp v0.7.0.

| Build | Same first result as F32 | Top-10 overlap with F32 | Mean cosine with F32, documents | Lowest cosine, documents |
|---|---|---|---|---|
| **VelociQ8 1.0** | **300 / 300** | **98.4 %** | 0.99987 | 0.99982 |
| Ampere llama.cpp v3.4.2 | 299 / 300 | 97.8 % | 0.99975 | 0.99967 |
| llama.cpp v0.7.0 + KleidiAI, Q8_0 | 299 / 300 | 96.7 % | 0.99944 | 0.99923 |
| llama.cpp v0.7.0, Q8_0 | 298 / 300 | 98.2 % | 0.99988 | 0.99982 |

"Same first result" counts the searches where the top document is the one F32 returns, which is what a search user sees. "Top-10 overlap" is the share of the F32 top 10 that the build also returns, averaged over the 300 searches.

On cosine, plain Q8_0 is as close to F32 as VelociQ8. On this machine VelociQ8 needs 47 % less time per document.

## LLMs

KL divergence against F32 logits, with `llama-perplexity --kl-divergence` on wikitext-2: 12 chunks of 512 tokens, 1 thread, on the Arm machine. Lower is better. "Same top token" is how often the most likely next token matches F32.

The VelociQ8 column was measured converting the BF16 model at load time. Pre-converted Q8A64 files give close values; the largest gap is on gemma-3-1b (KL divergence 0.001929, same top token 97.97 %).

These runs used the previous base, llama.cpp v0.6. The Q8A64 results do not depend on the base: VelociQ8 1.0 gives bit-identical outputs to that build on our control models.

| Model | VelociQ8 (Q8A64) | llama.cpp v0.6, Q8_0 | llama.cpp v0.6 + KleidiAI, Q8_0 | Same top token: VelociQ8 / + KleidiAI |
|---|---|---|---|---|
| Llama-3.2-1B | 0.000916 | 0.000950 | 0.009611 | 98.20 % / 95.03 % |
| gemma-3-1b | 0.001889 | 0.003248 | 0.034334 | 98.27 % / 90.78 % |
| Qwen3-1.7B | 0.002523 | 0.003306 | 0.032216 | 97.42 % / 91.44 % |
| granite-4.0-h-350m | 0.003113 | 0.006265 | 0.050487 | 96.11 % / 86.21 % |

On Llama-3.2-1B, VelociQ8 matches plain Q8_0. On the other three it has 24 % to 50 % less divergence. The KleidiAI build trades accuracy for speed: its divergence is 10 to 18 times higher than VelociQ8's.

### Mixture-of-experts LLMs

Same method, with three differences: the reference is the original F16 model run by llama.cpp v0.7.0, the runs use 8 chunks of 512 tokens and 4 threads, and the builds are VelociQ8 1.1 and llama.cpp v0.7.0. VelociQ8 converted the F16 model at load time.

| Model | VelociQ8 1.1 (Q8A64) | llama.cpp v0.7.0, Q8_0 | llama.cpp v0.7.0 + KleidiAI, Q8_0 | Same top token: VelociQ8 / Q8_0 / + KleidiAI |
|---|---|---|---|---|
| granite-3.1-1b-a400m | 0.00242 | 0.00363 | 0.01246 | 97.50 % / 97.06 % / 94.80 % |
| OLMoE-1B-7B | 0.00641 | 0.00748 | 0.01665 | 96.62 % / 96.72 % / 94.66 % |

VelociQ8 has 33 % less divergence than Q8_0 on granite and 14 % less on OLMoE. `llama-perplexity` reports a standard error of 0.35 to 0.40 points for each same-top-token rate. On OLMoE the rates of VelociQ8 and Q8_0 differ by 0.10 points; on granite, VelociQ8's is 0.44 points higher.

### Newer models

Same method against the original BF16 model: 8 chunks of 512 tokens, 4 threads, VelociQ8 converting BF16 at load time, against llama.cpp v0.7.0 with Q8_0. These runs used VelociQ8 1.0; 1.1 gives the same output on dense models.

| Model | VelociQ8 (Q8A64) | llama.cpp v0.7.0, Q8_0 | Same top token: VelociQ8 / Q8_0 |
|---|---|---|---|
| LFM2.5-1.2B-Instruct | 0.001720 | 0.001772 | 97.06 % / 97.30 % |
| Qwen3.5-0.8B | 0.000960 | 0.001241 | 98.33 % / 98.14 % |
| gemma-4-E2B-it | 0.004768 | 0.008783 | 97.01 % / 97.50 % |

On LFM2.5 the two are level. On Qwen3.5 VelociQ8 has 23 % less divergence, and on gemma-4 46 % less: there the largest single-token divergence was 0.107 for VelociQ8 and 4.583 for Q8_0.

### LFM3

Same method with 8 chunks of 512 tokens and 4 threads. The reference is the original BF16 model converted to F32, which loses nothing, run by llama.cpp b11963. VelociQ8 1.5 loaded the published Q8A64 files. The Q8_0 files are Liquid AI's own for LFM3-1.7B and LFM3-VL-2B, and made from the same BF16 with llama.cpp b11963 for the base model, which has no official GGUF. For LFM3-VL-2B the figures are for the language model on text; images were not measured.

| Model | VelociQ8 1.5 (Q8A64) | llama.cpp b11963, Q8_0 | Same top token: VelociQ8 / Q8_0 |
|---|---|---|---|
| LFM3-1.7B | 0.001487 | 0.001703 | 98.14 % / 98.09 % |
| LFM3-1.7B-Base | 0.000471 | 0.000546 | 99.07 % / 98.92 % |
| LFM3-VL-2B | 0.002580 | 0.002686 | 97.01 % / 97.11 % |

VelociQ8 has 13 % less divergence on LFM3-1.7B and 14 % less on the base model. On LFM3-VL-2B the two are level: the gap is 4 %, about the standard error `llama-perplexity` reports (0.000089). The standard errors of the divergence are 0.000074 and 0.000080 on LFM3-1.7B and 0.000013 and 0.000014 on the base model (VelociQ8, then Q8_0). The same-top-token rates differ by 0.05 points on LFM3-1.7B, 0.15 on the base model and 0.10 on LFM3-VL-2B, always under the standard error: 0.30 points on LFM3-1.7B, 0.21 to 0.23 on the base model and 0.37 to 0.38 on LFM3-VL-2B.
