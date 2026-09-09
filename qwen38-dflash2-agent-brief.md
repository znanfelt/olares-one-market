# Coding-Agent Brief: Add `llamacppqwen38dflashone`

## Mission

Implement one new, opt-in Olares Market chart named `llamacppqwen38dflashone`: the **speed-first DFlash2 sibling** to PR #19's stable, upstream-pure `llamacppqwen38mtpone` native-MTP chart.

This is not a rewrite of PR #19. Both charts must coexist:

| Chart | Tier | Backend | Default goal |
|---|---|---|---|
| `llamacppqwen38mtpone` | Stable | upstream `llama.cpp`, native MTP | reliable Q8-KV agentic coding baseline |
| `llamacppqwen38dflashone` | Fast / experimental | pinned `buun-llama-cpp` image or reproducible pinned buun build | DFlash2 + TurboQuant/VBR performance tier |

The DFlash chart deliberately follows the dense-model pattern of `llamacppqwen36dflashone`, while it reuses PR #19's Qwen3.8 target-model, shared-download, and fixed-chat-template logic. Use `llamacppqwen36a3bdflashone` only as a secondary reference for current upstream DFlash syntax; do not treat it as the primary implementation template.

## Required reading

Read these files and references before editing:

1. Root `CLAUDE.md` in full. It is authoritative for chart conventions, Olares runtime constraints, generated catalog artifacts, the hardware profile, and current shared-Hugging-Face environment conventions.
2. PR #19 / `llamacppqwen38mtpone` on branch `feat/add-qwen38-27b-mtp`: copy its Qwen3.8 target model, fixed template, `fix-shared-perms` init container, shared-model migration behavior, probes, Olares environment plumbing, and reasoning configuration.
3. `llamacppqwen36dflashone`: primary reference for the dense DFlash chart architecture, buun/TurboQuant settings, and manifest conventions.
4. `llamacppqwen36a3bdflashone`: secondary reference for DFlash download/idempotency and current upstream DFlash flag spelling only.
5. `Olares Market chart upload recipe.md` in the project workspace: use it for the separate custom-Market source-tree archive. It is not the same artifact as `helm package` output.

Do not modify PR #19. Do not edit an existing Qwen3.6 chart except where the repository's generated catalog process requires it.

## Fixed product decision

This chart is the **buun/TurboQuant DFlash2** implementation—not an open-ended upstream-versus-fork experiment.

Why:

- The motivating community configuration combines DFlash2 with buun's compressed KV path; an upstream Q8-KV configuration is a separate, worthwhile future comparison, not the same product.
- On a 24 GiB RTX 5090M, Qwen3.8 UD-Q4_K_XL plus Q8/Q8 KV at 131K already leaves very little headroom. A separate DFlash2 drafter adds weights, draft KV, and work buffers.
- buun provides TurboQuant/VBR options needed to create a realistic long-context DFlash tier, but it is experimental. The manifest and PR must state that plainly.

### Backend requirement

Before coding the deployment, resolve the exact reproducible backend **once**:

1. Verify that a published buun CUDA server image exists at the exact repository/namespace/tag you will use. Never assume `ghcr.io/spiritbuun/buun-llama-cpp` exists merely from the source repository name.
2. Pin a non-rolling image tag and record the immutable digest if the deployment/image tooling permits it. Never use `latest`, `master`, or an unpinned commit-derived tag.
3. Pull/run the candidate image, execute `/app/llama-server --help`, and confirm every selected flag is accepted: DFlash, draft model path, full draft GPU offload, the selected TurboQuant/VBR cache options, flash attention, and reasoning/template flags.
4. Verify the selected image is compatible with the host driver. Do not use a CUDA 13.3 image until `nvidia-smi` on `192.168.1.32` shows a driver at least 610.43.02; the currently documented working fallback is CUDA 12.
5. If no published, pin-able buun image can be verified, **stop before writing a chart and report the evidence**. Do not substitute an imagined image, silently use upstream, or publish an unpinned build. A separate decision can then authorize either an upstream DFlash2 chart or a reproducible custom-image build.

Source/provenance fields in the manifest and PR must name `spiritbuun/buun-llama-cpp`, its exact pinned image or source commit, the upstream lineage, and the DFlash2 model source. Do not present the chart as upstream-pure.

## Verified model inputs

Use these exact values unless verification against Hugging Face finds a current incompatible filename. Record the verification result in the PR.

```text
Target repository: unsloth/Qwen3.8-27B-GGUF
Target file:       Qwen3.8-27B-UD-Q4_K_XL.gguf
Draft repository:  incoai/Qwen3.8-27B-DFlash2-GGUF
Draft file:        Qwen3.8-27B-DFlash2-Q4_K_M.gguf
Draft fallback:    analogalok/Qwen3.8-27B-DFlash2-Q2_K-GGUF / corresponding Q2_K file
Chat template:     froggeric/Qwen-Fixed-Chat-Templates @ 855bffc49448e299789730ff92c9b8d834d6cc14
Template file:     chat_template.jinja
```

The Q4 target file is shared with the MTP chart. Preserve exactly the same `$SHARED_DIR` default (`/shared-models/llms`) and exact target filename so a user who already installed `llamacppqwen38mtpone` does **not** download another 17.6 GB copy. Use distinct `.ok` markers for the target and drafter. Verify repository/file existence before committing; do not guess Hugging Face paths.

## Memory-first launch profile

The MTP chart's Q8/Q8 KV profile at 131K consumed about 22.3 GiB before speculative overhead. Do **not** carry it unchanged into an external-DFlash chart.

Ship a conservative DFlash profile only after the selected buun binary has confirmed the exact cache flag syntax:

| Setting | Required starting policy |
|---|---|
| Target | UD-Q4_K_XL, full target GPU offload |
| Drafter | Q4_K_M; use verified Q2_K fallback if Q4 cannot coexist reliably |
| Context | Start at 96K. Promote 131K only after a stable on-device run; 200K is a benchmark target, not a manifest promise |
| KV cache | buun VBR constrained to a Turbo4 quality floor, or verified fixed Turbo4 K/V syntax. Do not use Q8/Q8 as the default |
| GPU layers | `99` target and `99` draft, or verified equivalent |
| HAMi limit | `CUDA_DEVICE_MEMORY_LIMIT_0=24000m` as a floor; this does not eliminate physical-VRAM limits |
| Slots | `--parallel 1` |
| Flash attention | Enabled |
| Reasoning | Fixed froggeric template; DeepSeek reasoning format; preserve reasoning |

Mandatory memory fallback order:

1. Q4 drafter + 96K + VBR/Turbo4.
2. Q2 drafter + 96K + VBR/Turbo4.
3. Reduce context to 64K.
4. Reduce batch size from 2048 to 512 before changing target quant.
5. Only then evaluate the documented target fallback `Qwen3.8-27B-UD-Q3_K_XL.gguf`; document every trade-off and measured result.

Do not promise 131K, 200K, 262K, or a performance number before measuring it on the Olares One.

## Required runtime semantics

Start from the selected buun image's verified `--help` output and write the server command using **only** flags that it accepts. The intended behavior is:

```text
--model "$SHARED_DIR/$TARGET_FILE"
--model-draft "$SHARED_DIR/$DRAFT_FILE"       # use only if verified spelling
--n-gpu-layers 99
--spec-draft-ngl 99                            # or verified equivalent; full draft offload is mandatory
--spec-type draft-dflash                       # or the image's verified DFlash2 spelling
--spec-draft-n-max 7
--spec-draft-p-min 0
--ctx-size 98304
--parallel 1
--flash-attn on
--jinja
--chat-template-file /config/chat_template.jinja
--reasoning-format deepseek
--reasoning-preserve
```

Do **not** copy `--reasoning off` from a Qwen3.6 DFlash chart. Qwen3.8 must retain the same fixed froggeric template and reasoning behavior as PR #19.

Use the selected buun cache mechanism only after verification. A conceptual VBR/Turbo4 profile is not permission to invent flags. Explicitly document whether the chart uses fixed Turbo4 or VBR, the quality floor, the cache budget behavior, and any feature restrictions such as session restore/context shifting.

Use the Qwen3.8 sampling defaults unless the runtime/model metadata proves a better compatible configuration:

```text
thinking:     temperature 1.0, top-p 0.95, top-k 20, min-p 0.0, presence penalty 0.0
non-thinking: temperature 0.7, top-p 0.80, top-k 20, min-p 0.0, presence penalty 1.5
```

The production depth is not predetermined. Start with `n-max=7`, then sweep `4`, `5`, `7`, and an optional `15` only if VRAM/stability permit. Keep the winner only if it improves sustained useful decode and reliability, not merely a short-run acceptance statistic.

## Chart implementation

Create the actual Helm/Olares chart only after the backend/image decision above is verified:

```text
llamacppqwen38dflashone/
├── Chart.yaml
├── OlaresManifest.yaml
├── values.yaml
├── owners
├── .helmignore
├── templates/
│   └── deployment.yaml
└── i18n/
    └── en-US/
        └── OlaresManifest.yaml
```

Do not put this brief inside the chart directory. Do not add a markdown implementation note to the packaged chart.

### Source templates and exact deltas

- Copy the Qwen3.8 MTP chart's `Chart.yaml`, manifest, values, owners, `.helmignore`, i18n file, and deployment as the starting structure.
- Merge only verified buun/TurboQuant/DFlash mechanics from `llamacppqwen36dflashone`.
- Retain Qwen3.8's `fix-shared-perms` init container, shared target migration, `emptyDir` `/config`, fixed-template download, health/startup/liveness probes, service exposure, security context, and Olares-specific volumes/env values.
- Add the drafter download using the same resumable `curl`, authorization-header, retry, and `.ok`-marker pattern as the existing DFlash sibling.
- Do not use `eval` with untrusted model values; preserve the repository's existing safe quoting pattern unless the copied implementation requires otherwise.

Use unique names consistently:

```text
ConfigMap: llamacpp-qwen38-dflash-env
Deployment: llamacppqwen38dflashone
Container: llamacpp-qwen38-dflash-server
Service: llamacppqwen38dflashone
Service port name: llamacpp-dflash
```

### Manifest rules

- `metadata.name` and `appid`: `llamacppqwen38dflashone`.
- Title: `Qwen38 27B DFlash`—it fits the permitted title character set and 30-character maximum.
- Keep the same standard boolean capability keys used by the dense existing DFlash chart. Use `mtp: false`; do **not** invent a nested `dflash` capability object.
- Put DFlash2/TurboQuant state in the existing `badge`, `hero.label`, specs, `upgradeDescription`, and `fullDescription` fields.
- Use `TBD t/s` for `bento.hero.value` until a measured on-device value exists.
- Declare every Olares environment value referenced by templates. In particular, follow the current `CLAUDE.md` and sibling-chart convention for optional `OLARES_USER_HUGGINGFACE_SERVICE` and `OLARES_USER_HUGGINGFACE_TOKEN`; values used but not declared can resolve empty.
- Preserve sufficient disk budget for target plus drafter; verify whether the copied 25Gi requirement is enough after the exact drafter file is chosen, and increase it if necessary.
- Do not add custom icons. Follow repository convention: generated icons and their catalog output are handled by the repository scripts.

## Required validation

### Local/repository validation — required before PR

```bash
helm lint llamacppqwen38dflashone/
helm template llamacppqwen38dflashone/
helm package llamacppqwen38dflashone/
npm run build:catalog
npm run dev
```

Confirm the rendered output includes ConfigMap, Deployment, and Service. Confirm the generated catalog contains the new eight-character hex appid and chart payload. Do not commit generated `.tgz` files when this repository ignores them; commit the changed generated source artifacts required by the Worker, including `src/catalog.json`, `src/charts.json`, and `src/icons.json` if changed. Check `git status` deliberately before committing.

If preparing a separate custom-Market upload archive, build it as a plain source-tree `.tar.gz`, with exactly one top-level `llamacppqwen38dflashone/` directory and its `Chart.yaml` directly beneath it. Do not upload a wrapper directory or a packaged Helm `.tgz`.

### On-device validation — required before publishing benchmarks

Target host: `ssh olares@192.168.1.32`.

1. Confirm driver/image compatibility and the exact GPU memory limit.
2. Install the chart and inspect logs through target and draft download, template fetch, model loading, full target/draft GPU offload, cache initialization, and server readiness.
3. Verify `GET /health`, a normal chat response, a thinking-mode response, a non-thinking response, and one structured tool-call round trip. Check that raw `<think>` tokens do not leak into ordinary content.
4. Use matched A/B testing: target-only baseline versus DFlash2, same target quant, context, prompt, sampling, output cap, batch settings, and hardware conditions.
5. Discard one cold generation after each server load. Run at least three warm 512-token completions per configuration; restart between configurations when the cache/back-end profile changes.
6. Record prompt throughput, decode throughput, drafted tokens, accepted tokens, acceptance rate/mean acceptance length, peak VRAM, context, selected KV mode, image digest, template revision, exact command, and errors/restarts.
7. Run the repository's `Space Invaders HTML, 2000 tokens` benchmark for at least ten trials and report average/min/max sustained decode separately from short completion results.
8. Sweep DFlash `n-max` at 4, 5, 7, and optionally 15. Sweep at least 64K, 96K, and 131K context only when prior points are stable.
9. Run an agentic tool-use/recovery test: plan, tool call, substantial tool output, deliberate tool error, correction, and final response. Document tool-call integrity, loop behavior, reasoning behavior, and stability.

No desktop/cloud/forum number is valid evidence for the manifest or PR. Never substitute model-card, Reddit, or community figures for measurements from this Olares One.

## PR and commit requirements

Use an imperative commit subject. For example:

```text
llamacppqwen38dflashone v1.0.0: add Qwen3.8 DFlash2 chart
```

Do not add a token/s result until it is measured. Follow the repository's normal body and `Co-Authored-By:` convention when applicable.

The PR must:

- Describe the chart as the experimental fast sibling to PR #19—not a replacement for its stable upstream MTP tier.
- Identify the exact buun image tag/digest or reproducible build commit, source repository, and why the fork is required.
- Identify target, drafter, chat-template revision, final flags, KV policy, context, resource limits, and whether Q2 drafter fallback was needed.
- Include a VRAM budget and the fallback path used.
- Include matched baseline/DFlash tables, depth sweep, context sweep, and tool-use/reliability outcomes only if tested on device.
- Otherwise explicitly say on-device installation and benchmark results are pending, keep `TBD t/s`, and do not claim unmeasured speed.
- State that the raw packaged `.tgz` is intentionally ignored and that regenerated Worker catalog artifacts are committed.

## Final checklist

- [ ] Brief remains outside the chart directory.
- [ ] Branch is based on current `main`.
- [ ] Exact buun CUDA image/tag/digest or reproducible source commit was verified before chart creation.
- [ ] Selected binary accepts every DFlash, cache, offload, template, and reasoning flag.
- [ ] No CUDA 13.3 image is used on the pre-610.43.02 host driver.
- [ ] Exact target and draft files exist and the target reuses the MTP chart's shared path/file.
- [ ] Q4 drafter → Q2 drafter → 64K context → smaller batch → Q3 target is the documented OOM fallback order.
- [ ] Full draft GPU offload is explicitly enabled.
- [ ] DFlash starts at n-max 7 and has a documented 4/5/7 sweep.
- [ ] Reasoning is not copied as off from the Qwen3.6 template.
- [ ] Only existing manifest capability keys are used.
- [ ] All referenced Olares variables are declared in `envs:`.
- [ ] Image, target model, drafter, and template are pinned/verified; no floating tags.
- [ ] Helm rendering and package validation pass.
- [ ] Catalog artifacts needed by the Worker are regenerated and staged; ignored `.tgz` artifacts are not committed.
- [ ] Custom upload archive, if made, has the required single-level source-tree structure.
- [ ] `TBD t/s` remains until verified on-device benchmark data exists.
