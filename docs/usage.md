# Usage

VelociQ8 ships the standard llama.cpp tools: `llama-server`, `llama-cli`, `llama-embedding` and `llama-bench`, since 1.1 `llama-quantize`, and since 1.2 `llama` (the single command that the Llama app for Windows runs) and `llama-tts`. Options, environment variables and the HTTP API are the ones documented by the llama.cpp release each version is based on (b11738 for 1.2). This page covers only what VelociQ8 changes.

## Loading a model

There are two ways to run a model in Q8A64.

**From a BF16, F16 or F32 GGUF.** VelociQ8 converts the weights to Q8A64 while it loads the model. This is on by default; nothing has to be enabled. With `-lv 5` (or `--verbose`), the load log shows a `CPU_Q8A64` buffer:

```text
load_tensors:   CPU_Mapped model buffer size =   384.00 MiB
common_memory_breakdown_print: |   - CPU_Q8A64   |   99 = 99 + 0 + 0 |
```

Start from the BF16 or F16 release of a model, not from a Q8_0 or Q4 file. Weights that are already quantized stay in their own format and run on the normal llama.cpp kernels. Since 1.1 this includes the expert weights of mixture-of-experts models.

**From a pre-converted Q8A64 file.** These GGUF files already contain the packed weights, so loading skips the conversion. With `-lv 5`, `print_info` reports them as `file type = packed CPU`. Only VelociQ8 can load them, and files of mixture-of-experts models need VelociQ8 1.1. Ready-made files are on [Hugging Face](https://huggingface.co/VelociQ8); to write your own, see [Converting a model](#converting-a-model).

To compare with the normal llama.cpp kernels, load the BF16, F16 or F32 GGUF with `--no-repack`: the conversion is skipped. Pre-converted Q8A64 files cannot run without Q8A64, so with `--no-repack` they fail to load.

## Converting a model

Since 1.1 the archives include `llama-quantize`, which writes a Q8A64 file from a BF16, F16 or F32 GGUF:

```sh
./llama-quantize model-BF16.gguf model-Q8A64.gguf Q8A64 4
```

The last argument is the number of threads. Q8A64 takes only these four arguments: input, output, type and threads. The other `llama-quantize` options (importance matrix, per-tensor types and so on) do not apply to it. Results are very close to converting the same model while it loads (see [Quality](quality.md)), and only VelociQ8 can load the file. Files of mixture-of-experts models need VelociQ8 1.1.

Converting needs memory for the input model and the result. A pre-converted file is worth it for models you load often or that are large: it loads faster and avoids the memory peak of converting at load time (see [Compatibility](compatibility.md)).

## Threads

Use `-t` with the number of physical cores you want to give the model. The Q8A64 kernels use the same thread pool as the rest of llama.cpp.

## Memory locking

`--load-mode mlock` and `--load-mode mmap+mlock` keep the model in RAM so the system cannot swap it out. They work with Q8A64 on Linux and Windows. On Linux, raise the locked-memory limit first (`ulimit -l`, or `LimitMEMLOCK=` in a systemd unit). If the limit is too low, the log shows `failed to mlock` and the model keeps running without the lock.

## Embeddings server

```sh
./llama-server -m model-BF16.gguf --embedding -t 4 -b 2048 -ub 2048 --port 8080
```

Each input must fit in one batch. With the default `-ub` of 512, longer texts are rejected with `input (N tokens) is too large to process`; `-b 2048 -ub 2048` accepts up to 2,048 tokens. Raise both for longer texts, up to the context of the model; memory use grows with them.

The endpoints are `/v1/embeddings` (OpenAI-compatible) and `/health`. The release builds do not include the web chat interface; the HTTP API is complete.

Use the prompt prefixes your embedding model expects. EmbeddingGemma, for example, expects `task: search result | query: ` for queries and `title: none | text: ` for documents.

## Ollama

The `velociq8-1.2-ollama-0.40.8-*` archives replace the llama.cpp server that Ollama 0.40.8 runs on the CPU with the VelociQ8 one, so Ollama can load Q8A64 files and convert BF16, F16 and F32 models to Q8A64 while loading. Ollama itself is not modified, and the archive works only with that Ollama version.

Stop Ollama and copy the contents of `lib/ollama/` from the archive over the `lib/ollama` folder of the installation: `/usr/local/lib/ollama` with the Linux install script, `%LOCALAPPDATA%\Programs\Ollama\lib\ollama` on Windows. Then import the model:

```sh
printf 'FROM ./embeddinggemma-2-Q8A64.gguf\n' > Modelfile
ollama create embeddinggemma-2-q8a64 -f Modelfile
```

Embedding models work with `/api/embed` and chat models with `/api/chat` and `/api/generate`. Q8A64 files published before 1.2 lack the two header fields Ollama needs to detect embedding models; the files on Hugging Face have been updated, and `llama-quantize` from 1.2 writes them. Reinstalling or updating Ollama restores its own server. The Windows archive has been checked but not yet run on Windows.

## Llama app for Windows

The [Llama app](https://github.com/ggml-org/Llama-Windows) uses a `llama.exe` it finds in `PATH` and leaves it alone. Add the folder of the VelociQ8 Windows archive to `PATH` before starting the app, and it will run VelociQ8 and load Q8A64 files. For embedding models, add `--embeddings` to the app's custom server arguments.

## Docker

The image `ghcr.io/dhaern/velociq8:1.2` runs `llama-server` from the release archives, on the CPU, for `linux/arm64` and `linux/amd64`. On Arm it contains the `linux-arm64-multi` build, so it picks the best kernels for the CPU at startup. On x86-64 it needs AVX2 and FMA.

The server listens on port 8080 inside the container and runs as user 10001. The image does not download models: mount a folder with your GGUF files on `/models`, readable by user 10001. Everything after the image name is passed to `llama-server`:

```sh
docker run --rm -p 127.0.0.1:8080:8080 \
  --mount type=bind,src=/path/to/models,dst=/models,readonly \
  ghcr.io/dhaern/velociq8:1.2 \
  -m /models/embeddinggemma-2-Q8A64.gguf --embedding -t 4 -b 2048 -ub 2048
```

The container reports itself healthy once `/health` answers.

## HTTPS

The release builds have no HTTPS support. `llama-server` serves plain HTTP only and exits with an error if both `--ssl-key-file` and `--ssl-cert-file` are given; with only one of them it ignores it and keeps serving plain HTTP. The tools cannot download models over HTTPS either, so options such as `-hf` do not work: download the GGUF file first and pass it with `-m`. If clients need TLS, put a reverse proxy in front of `llama-server`.

This `compose.yaml` puts Caddy in front of the Docker image. Caddy obtains and renews a certificate for your domain, and `llama-server` is reachable only through it:

```yaml
services:
  velociq8:
    image: ghcr.io/dhaern/velociq8:1.2
    command: ["-m", "/models/${MODEL_FILE:?Set MODEL_FILE to the GGUF filename}", "-t", "${THREADS:-4}", "-c", "2048", "-ngl", "0"]
    environment:
      LLAMA_API_KEY: "${API_KEY:?Set API_KEY}"
    volumes:
      - "${MODELS_DIR:-./models}:/models:ro"
    restart: unless-stopped
  caddy:
    image: caddy:2-alpine@sha256:d8542f48d34a9cf4e4c11a478865229840e87e4c96ea3f439101f31a5d35f75f
    depends_on:
      velociq8:
        condition: service_healthy
    environment:
      DOMAIN: "${DOMAIN:-localhost}"
    ports:
      - "${BIND_ADDRESS:-0.0.0.0}:${HTTP_PORT:-80}:80"
      - "${BIND_ADDRESS:-0.0.0.0}:${HTTPS_PORT:-443}:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy_data:/data
      - caddy_config:/config
    restart: unless-stopped
volumes:
  caddy_data:
  caddy_config:
```

Next to it, a `Caddyfile`:

```text
{$DOMAIN:localhost} {
    reverse_proxy velociq8:8080
}
```

Put the models in `./models` (or set `MODELS_DIR`) and start both services:

```sh
export API_KEY="$(openssl rand -hex 32)"
DOMAIN=llm.example.com MODEL_FILE=granite-3.1-1b-a400m-instruct-f16.gguf docker compose up -d
```

Without `API_KEY`, `docker compose` refuses to start. Keep the key: clients send it in the `Authorization` header; requests without it get a 401 error. `/health` does not need it:

```sh
curl https://llm.example.com/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

The domain must point to the host, and ports 80 and 443 must be reachable from the internet so Caddy can get the certificate. For an embedding model, add `"--embedding", "-b", "2048", "-ub", "2048"` to `command`. To try the setup without a domain, write `localhost` and `tls internal` in the `Caddyfile`:

```text
localhost {
    tls internal
    reverse_proxy velociq8:8080
}
```

Caddy then signs the certificate with its own local authority, which clients do not trust by default (use `curl -k` for a quick test).
