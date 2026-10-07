# Plan: Phase B / C Development

- **Date**: 2026-10-07
- **Status**: proposed
- **Scope**: Phase B (`QuantAssessment`, weight reconstruction error) and Phase C (`EvalRecord`, PPL / KLD ingest), as defined in the [README](../../README.md)
- **Inputs**:
  - this repo's README (report models, methodology, storage, roadmap): the README defines *what*, this plan defines *how and in what order*
  - ModelSpec plan [Quantization Quality Assessment](https://github.com/parapulseio/ModelSpec/blob/main/docs/plans/2026-10-07-quantization-quality-assessment.md), sections 4, 5 (original design) and 9 (ModelSpec-side prerequisites)

## 1. Goals and non-goals

**Goals**

- Reproducible weight reconstruction error for GGUF / AWQ / GPTQ against the original weights, fetching only the sampled tensor bytes.
- Importing PPL / KLD results with mandatory comparability metadata, and only ever comparing like with like.
- Versioned report models with published JSON Schemas that ParaPulse can ingest.

**Non-goals**

- Running models, computing PPL / KLD, or needing a GPU.
- A results database or leaderboard UI. Leaderboards come after the report format is stable.
- Re-implementing what ModelSpec already does (repo resolution, lineage, header parsing, quantization metadata).

## 2. Design principles

These mirror ModelSpec so that both repos read the same way:

1. **Pydantic only at the boundaries.** Report models validate on entry (`ingest`) and serialize on exit. Fetching, unpacking, dequantization and metrics are plain Python + numpy and never import the report models.
2. **Refuse rather than guess.** Unsupported packing, unmappable tensors and shape mismatches produce an explicit skip with a reason in the report, never a number.
3. **Shape assertion before any metric.** A silent transpose or permutation produces plausible-looking but meaningless numbers.
4. **Never download full weights by default.** Every fetch is a Range request for one tensor. Wider coverage (`--all-layers`) prints a byte estimate first.
5. **Never run a model.** Phase C is ingest only.

## 3. Decisions

| ID | Decision | Status |
|---|---|---|
| D2 | Phase B / C live in this repo, not in ModelSpec. Storage as in the README: stdout / `-o`, sidecar in a ModelSpec `--download-only` directory, ParaPulse for persistence. | decided (ModelSpec plan D2) |
| D3 | HF ↔ GGUF handling is split into **names** and **values** (section 6, B6). | **proposed, needs confirmation** |
| D4 | Phase C is scheduled independently of the Phase B prototype gate (section 9). | **proposed, needs confirmation** |

## 4. Upstream prerequisites in ModelSpec

These are tracked in section 9 of the ModelSpec plan and block specific tasks here:

| ID | Change in ModelSpec | Blocks |
|---|---|---|
| U1 | Public, backward-compatible GGUF tensor-info API returning `(name, dims, ggml_type, offset)` plus the data-section start (end of tensor infos aligned to `general.alignment`, default 32). Today `parse_gguf_header` reads the offset and discards it. | B2 (GGUF fetch) |
| U2 | A release tag (e.g. `v0.2.0`) containing Phase A + U1. The latest tag, `v0.1.1`, predates Phase A. | B0 dependency pin, B8 |

Until U2 exists, B0 can pin a ModelSpec commit SHA and switch to the tag when it lands.

What this repo uses from ModelSpec:

- `modelspec.pipeline.extract`: `identity.lineage.base_models` (reference discovery), `architecture.family`, `attention.num_heads` / `num_kv_heads` (inverse permutation), `quantization.*` (format, bits, group size, `desc_act`, quantized-module lists).
- U1 for GGUF tensor offsets.

What this repo owns itself:

- Range requests at arbitrary offsets. ModelSpec's fetch helpers are private and only read prefixes.
- Commit SHA resolution (`huggingface_hub.HfApi.model_info(...).sha`). ModelSpec records it in its download manifest, not in `ModelSpec`.
- Fields ModelSpec does not model, read from `config.json` / `quantize_config.json` directly: `checkpoint_format` (`gptq` / `gptq_v2`), AWQ `version` (GEMM / GEMV), packing backends.

## 5. Package layout

```
quantassess/
├── schema/          # Pydantic report models: QuantAssessment, EvalRecord; JSON Schema export
├── fetch/           # Range-request tensor fetch: safetensors (+ index, BF16), GGUF (via U1)
├── formats/         # unpack + dequantize: awq.py, gptq.py, gguf.py; packing detection
├── mapping/         # HF ↔ GGUF names (gguf-py TensorNameMap) + value-transform table
├── metrics.py       # rel. error, SQNR, cosine, max abs error, imatrix-weighted variants
├── sampling.py      # default sample set, --all-layers, byte estimate
├── assess.py        # Phase B orchestration: resolve → map → fetch → dequant → compare → report
├── ingest/          # Phase C: llama_perplexity.py, hf_json.py, comparability checks
├── compare.py       # like-for-like comparison helpers over reports / records
└── cli.py           # quantassess assess | ingest | schema
tests/               # parallel structure; no network by default
```

Dependencies: `pydantic>=2`, `numpy`, `requests`, `huggingface_hub`, `gguf>=0.19` (TensorNameMap coverage and dequantizers), and `modelspec` from git. Optional extra `plot = ["matplotlib"]`. Versioning follows ModelSpec: `hatch-vcs`, git-tag-derived, no static version string.

## 6. Phase B tasks

### B0. Skeleton

- `pyproject.toml` (hatch + hatch-vcs), `quantassess/` package, CLI entry point, `pytest` config, CI workflow, `AGENTS.md`.
- Pin `modelspec` (commit SHA until U2, then the tag) and `gguf`.
- **Done when**: `pip install -e ".[dev]"` and `pytest -q` pass on CI, and `quantassess --version` prints a git-derived version.

### B1. Report models and JSON Schema

- `QuantAssessment` and `EvalRecord` exactly as in the README tables, each with a `schema_version`.
- Subject and reference both carry `repo_id` + commit SHA (required) and an optional file.
- `EvalRecord` validation rejects records with incomplete comparability metadata.
- `quantassess schema assessment|eval-record` prints a self-identifying JSON Schema (`$schema`, `$id`, generator version), the same approach as ModelSpec's `export_json_schema()`.
- **Done when**: schema tests feed dicts and assert accept / reject; the exported schemas validate example documents.

### B2. Tensor fetch

- **safetensors**: 8-byte length → JSON header → `data_offsets` slice; shard lookup via `model.safetensors.index.json`; BF16 → FP32 by bit shift; guard against CDNs that ignore `Range` (cap the bytes read, as ModelSpec's `_read_prefix` does).
- **GGUF** (needs U1): tensor offset + data-section start, size from `GGML_QUANT_SIZES`, Range-fetch the slice.
- `HF_TOKEN` support for gated repos; byte accounting for the report's cost section.
- **Done when**: unit tests with synthetic files served from a local HTTP server (or a fake session) cover single-file, sharded, BF16 and GGUF; one opt-in network test fetches one real tensor.

### B3. AWQ dequant (GEMM only)

- Unpack with reverse order `[0,4,1,5,2,6,3,7]`, `w = (q − z) · s`, transpose to `[out, in]`.
- Detect GEMV / other packings from `quantization_config` and refuse them.
- **Done when**: round-trip test (synthetic matrix → pack in AWQ GEMM layout → load → compare) passes, and a GEMV config is refused with a clear error.

### B4. GPTQ dequant (v1, v2, `g_idx`)

- v1: `w = (q − (z + 1)) · s`; v2 (`checkpoint_format: gptq_v2`): no `+1`; `desc_act=True`: `scale[g_idx[row]]`.
- Refuse Marlin / ExLlama-repacked checkpoints.
- **Done when**: round-trip tests pass for all three variants, plus a test showing that v1 vs v2 detection changes the result by exactly one quantization level.

### B5. GGUF dequant

- `gguf.quants.dequantize(bytes, qtype)` on the fetched slice; never `GGUFReader`.
- **Testing constraint (verified on gguf 0.19.0)**: gguf-py can *quantize* only F32 / F16 / BF16 / Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 / TQ1_0 / TQ2_0 / MXFP4, but can *dequantize* all K-quants and IQ types. Round-trip tests are therefore possible only for the simple types. For K / IQ types, check in small fixtures produced by `llama-quantize` together with the FP32 source, and assert the error is within the expected band.
- **Done when**: round-trip tests for Q4_0 / Q8_0 pass, and fixture tests for Q4_K / Q6_K / one IQ type pass.

### B6. HF ↔ GGUF mapping (decision D3, proposed)

Names and values are handled separately:

- **Names, all architectures**: `gguf.get_tensor_name_map(arch, n_blocks)`. `arch` comes from GGUF `general.architecture`, `n_blocks` from `{arch}.block_count`. gguf 0.19.0 covers 126 architectures; spot checks for Llama, Qwen2, Phi3, Gemma2 and DeepSeek2 map the dense tensors correctly. We maintain no name table of our own. The gguf version is recorded in the report.
- **Values, an explicit transform table, default deny**:

  | Transform | Applies to | Status |
  |---|---|---|
  | identity | dense `down/up/gate/v/o_proj` on most architectures | first version |
  | inverse q/k row permutation | Llama-family `attn_q` / `attn_k` (needs head counts from ModelSpec) | second step |
  | MoE expert slice | per-expert HF tensors, e.g. `experts.0.w2`, which have no GGUF name: gguf stacks all experts into one 3-D tensor such as `ffn_down_exps` | second step |
  | anything else (architecture-specific split / transpose in llama.cpp's converter) | — | not supported |

- **Fallback**: a tensor with no registered transform is compared at the distribution level only (both sides sorted independently), and the report says so.
- **Self-check**: if element-wise cosine is low while the distribution-level comparison is close, flag "suspected layout mismatch" instead of reporting poor quality. This keeps a missing transform from turning into a wrong conclusion.
- AWQ / GPTQ need no mapping: they keep HF names.
- **Done when**: tests cover name mapping for several architectures, identity and fallback modes, and the self-check firing on a deliberately permuted tensor.

### B7. Metrics, sampling, aggregates

- Relative error, SQNR (dB), cosine, max absolute error; aggregates by tensor role and by layer.
- Default sample: `mlp.down_proj` + `self_attn.v_proj` at the first, middle and last layer. `--all-layers` prints a byte estimate first.
- Skip unquantized tensors (norms, embeddings, usually `lm_head` in AWQ / GPTQ), using ModelSpec's quantized-module lists, and record the reason.
- **Done when**: metric tests on hand-computed matrices pass, and the default sample for a known layout matches the expected tensor list.

### B8. `assess` orchestration and CLI

- Reference resolution: ModelSpec `identity.lineage.base_models` (single base → use it; several or none → require `--base`), commit SHAs for subject and reference.
- Output: stdout, `-o`, or the sidecar path `./<repo_id>/assessments/<reference>__<method-hash>.json` when the input is a ModelSpec `--download-only` directory.
- `--plot` (plot extra): sorted step plot of one tensor.
- **Done when**: an end-to-end test with fake fetch sessions produces a schema-valid report, and the CLI help matches the README's usage section.

### B9. Prototype gate

- One real run on a ≤ 1B model with GGUF Q4_K_M + AWQ + GPTQ variants, < 200 MB transferred in default sampling mode.
- Candidate: Qwen2.5-0.5B-Instruct (to verify: official AWQ / GPTQ-Int4 / GGUF variants exist, and Qwen2 does not permute q/k in GGUF). An alternative is Llama-3.2-1B, which is gated and needs third-party AWQ / GPTQ variants.
- Write the results and observations into `docs/` and decide **keep / extend / drop** with ParaPulse.

### B10. imatrix-weighted error (after the gate, only if kept)

- `--imatrix path`; `importance = in_sum2 / counts` per input column; GGUF names, so it reuses B6.
- Warn when the imatrix is the one used to produce the assessed quant.

## 7. Phase C tasks

C depends only on B1 (report models). It does not touch fetch, dequant or mapping.

### C1. `llama-perplexity` parser

- Parse plain PPL output (`Final estimate: PPL = x ± y`) and `--kl-divergence` output (PPL quant / base, mean ln ratio, KLD mean / median / 99% / 99.9% / max, mean Δp, RMS Δp, same-top-p).
- Output wording changes between llama.cpp builds: parse tolerantly, always keep the raw text, and fail loudly on a missing required metric.
- **Done when**: tests pass on checked-in real logs from at least two llama.cpp builds.

### C2. HF-eval JSON format

- Document a JSON format for transformers-side evaluations, plus a reference script in `docs/` (not run in tests). The script must record its scoring scheme (all positions vs second half).
- **Done when**: format documented, example file validates.

### C3. `ingest` CLI and comparability checks

- `quantassess ingest <file> --subject <repo>@<sha> --base <repo>@<sha> --dataset ... -c ... --chunks ...`; reject records with missing comparability metadata.
- **Done when**: CLI tests accept a complete record and reject each kind of missing metadata.

### C4. Like-for-like comparison helpers

- Compare records only when tool, dataset (+ hash), `-c`, `--chunks`, scoring scheme and reference all match. Otherwise explain which fields differ.
- **Done when**: tests show a llama.cpp record and a transformers record are never ranked together.

## 8. Dependency graph

```
  U1 --------------------------+
                               v
  B0 ---> B1 ---+---> B2 ---> B8 ---> B9 (gate) ---> B10
                |      ^
                |      |
                +---> B3, B4, B5, B6, B7  (parallel)
                |
                +---> C1, C2 ---> C3 ---> C4
```

- B2–B7 depend only on B1 (and B0) and can run in parallel. B8 integrates them. B2's GGUF half waits on U1; the safetensors half does not.
- U2 (ModelSpec release) is needed before B8 pins a tag; until then, pin a commit.
- Critical path: B0 → B1 → B2 → B8 → B9.

## 9. Staffing (3 developers)

```
        step 1        step 2             step 3
dev 1:  B0, B1  --->  B2 ------------->  B8 ---> B9
dev 2:                B3, B4, B5  --->   (help B8)
dev 3:  U1 (ModelSpec) -> B6, B7 ---->   C1 ... C4
```

Phase C fits on dev 3 after B6 / B7 (decision **D4**, proposed). If D4 is rejected, Phase C waits for the B9 gate instead, as originally planned in ModelSpec.

## 10. Testing strategy

- `pytest -q` with no network by default; network tests are opt-in (marker + env var).
- Every dequant path has a round-trip test or, for K / IQ types, a llama.cpp-generated fixture (B5).
- Fixtures are tiny: a few blocks per tensor, never real model weights.
- Report examples in `tests/fixtures/` validate against the exported JSON Schemas.

## 11. Risks

| Risk | Mitigation |
|---|---|
| Packing variants (GPTQ v2, GEMV, Marlin, …) silently produce wrong numbers | detect from config, refuse unknown packings, round-trip tests |
| Missing value transform for an architecture | default deny, distribution-level fallback, layout-mismatch self-check (B6) |
| llama.cpp output format drift | tolerant parser, raw text kept, multi-build fixtures (C1) |
| ModelSpec API churn | pin commit / tag; depend only on public API plus U1 |
| Weight error read as a quality verdict | the README and report say "diagnostic, rank by KLD"; aggregates never output a single "score" |
| Prototype shows the numbers are not useful | B9 gate before B10 and leaderboards |

## 12. Open questions

- Confirm **D3** (name / value split, default-deny transforms).
- Confirm **D4** (Phase C in parallel vs after the gate).
- Prototype model for B9: Qwen2.5-0.5B-Instruct or Llama-3.2-1B?
- Should the report carry an optional ModelSpec snapshot of the subject (selected fields only), or only `repo_id` + SHA?
