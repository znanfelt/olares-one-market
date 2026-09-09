# Agent Brief: `llamacppqwen38dflashone` — Qwen3.8-27B + DFlash2 speculative decoding (upstream llama.cpp)

## Context you must read before writing any code

This repo is `aamsellem/olares-one-market`, a Cloudflare-Worker-backed custom Olares Market source. Read `CLAUDE.md` at repo root in full before starting — it documents the repo layout, the Olares manifest constraints, the battle-tested llama.cpp flags for the RTX 5090M (24GB), and the hardware profile (Intel Core Ultra 9 275HX, 96GB DDR5, no AVX-512/AMX, sm_120 Blackwell GPU).

This task is the direct, planned follow-up to PR #19 (`llamacppqwen38mtpone`, branch `feat/add-qwen38-27b-mtp`). That PR shipped a **native-MTP** chart for Qwen3.8-27B on **upstream, unmodified** `ghcr.io/ggml-org/llama.cpp:server-cuda12-b10666`. Its own "Known limits & follow-ups" section explicitly named a DFlash2 sibling chart as the next step, and a PR comment from `bayerhazard` reported 80–110 t/s with a DFlash2-based stack versus the MTP chart's own ~50 t/s. **Do not modify or reopen PR #19.** This is a new, separate chart and a new PR.

## Design intent (read this, then follow it exactly)

Build a **second, sibling chart**, `llamacppqwen38dflashone`, positioned exactly the way `llamacppqwen36a3bdflashone` is positioned relative to `llamacppqwen36mtpone`:

- `llamacppqwen38mtpone` (existing, PR #19) = stable / upstream-pure / native-MTP tier.
- `llamacppqwen38dflashone` (this task) = fast / DFlash2 speculative-decoding tier.

Keep the two charts independent Helm charts with independent `Chart.yaml`, `OlaresManifest.yaml`, `appid`, k8s object names, and ports collision avoidance is not needed (each Olares app gets its own namespace) but resource names inside the chart must be unique to this chart (do not reuse `llamacppqwen38mtpone` as any k8s object name).

### Backend decision — attempt upstream first, fall back to fork

1. First, check whether upstream `ghcr.io/ggml-org/llama.cpp` server-cuda12/server-cuda13 images at a build number from August–September 2026 or later expose `--spec-type draft-dflash` (this flag already exists in the repo's `llamacppqwen36a3bdflashone` chart for the Qwen3.6 DFlash sibling, so the flag itself is proven upstream — the question is only whether it works correctly for the Qwen3.8-27B target model + a DFlash2 drafter GGUF).
   - Pull a recent tag, e.g. `ghcr.io/ggml-org/llama.cpp:server-cuda12-b10800` or later (check https://github.com/ggml-org/llama.cpp/pkgs/container/llama.cpp for the latest tags past b10666), and confirm `--help` lists `draft-dflash` as a valid `--spec-type` value.
   - Verify a DFlash2 drafter GGUF exists for Qwen3.8-27B compatible with mainline llama.cpp's DFlash loader (needs `dflash.target_layers`-style metadata, same requirement noted for the Qwen3.6 DFlash drafter in `llamacppqwen36a3bdflashone/templates/deployment.yaml`). Candidates to check on Hugging Face: `incoai/Qwen3.8-27B-DFlash2-GGUF` (Q4_K_M ~1.1 GB) and any EXL3/GGUF conversions referenced from `z-lab/Qwen3.8-27B-DFlash2`. Confirm via `curl -s -o /dev/null -w "%{http_code}" "https://huggingface.co/api/models/<org>/<repo>/tree/main?recursive=1"` per CLAUDE.md's HF verification pattern.
2. If upstream `--spec-type draft-dflash` works cleanly with a Qwen3.8 DFlash2 GGUF drafter on a pinned `server-cuda12-b*` or `server-cuda13-b*` tag: use it. Stay upstream-pure, matching the MTP chart's own "no fork, no custom image" philosophy, and pin the exact build number in the image tag.
3. If it does not work (loader error, missing metadata support, crash on the drafter GGUF), fall back to `ghcr.io/spiritbuun/buun-llama-cpp` (the experimental fork referenced by the community commenter) — but only after confirming this fork actually publishes a container image and documenting the specific tag you use. If you fall back to the fork, note this explicitly and loudly in the PR body as a deliberate, justified deviation from the "upstream-pure" pattern (do not silently pick the fork).
4. Whichever backend you land on, do not use the driver-blocked `server-cuda13` family unless you have first confirmed on the Olares One host (`ssh olares@192.168.1.32`, per CLAUDE.md) that the NVIDIA driver has actually been bumped past 595.84 (see `beclab/Olares#4100`, driver 610.57.04). If unconfirmed, stay on `server-cuda12`.

### Model and quantization

- Target model: `unsloth/Qwen3.8-27B-GGUF`, file `Qwen3.8-27B-UD-Q4_K_XL.gguf` (~17.6 GB) — same target as the MTP chart, for a clean apples-to-apples comparison. Do not switch quant tiers unless VRAM math (below) forces it.
- Drafter: the smallest available Qwen3.8-27B DFlash2 GGUF drafter (target ~1.1 GB, e.g. `incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M`). Verify the repo/file exist via the HF HEAD-check pattern before wiring it into the chart.
- Chat template: reuse the exact same pinned template as the MTP chart — `froggeric/Qwen-Fixed-Chat-Templates` @ commit `855bffc49448e299789730ff92c9b8d834d6cc14` (v22.5) — fetched once at container start into an `emptyDir` at `/config`, identical mechanism to `llamacppqwen38mtpone/templates/deployment.yaml`. Do not re-derive this; copy the fetch logic.

### Runtime flags — start here, then sweep

Base everything on `llamacppqwen36a3bdflashone/templates/deployment.yaml` (the DFlash sibling pattern) merged with `llamacppqwen38mtpone/templates/deployment.yaml` (the Qwen3.8-specific fetch/template logic):

- `--spec-type draft-dflash` (or the fork equivalent) with `--spec-draft-n-max` starting at **7** (the DFlash2 design point — block size 8, 7 draft tokens per verification step; do not copy the MTP chart's n_max=2, that value is specific to MTP, not DFlash2).
- `--cache-type-k q8_0 --cache-type-v q8_0` as the safe starting KV config (matches the MTP chart). Only move to a fork-specific codec (e.g. `turbo4`/VBR) if you have confirmed the backend actually supports it — do not invent flags.
- `--ctx-size 131072` as the starting context (same as MTP chart). Note in the PR that 200K was community-reported feasible with a different KV codec, but do not claim that on-device until measured.
- `--n-gpu-layers 99`, `--flash-attn on`, `--jinja`, `--batch-size 2048 --ubatch-size 512`, `--threads 16 --threads-batch 16`, `--parallel 1` — same as MTP chart.
- `--reasoning-format deepseek --reasoning-preserve` — same as MTP chart (reasoning ON, matching the MTP chart's departure from the Qwen3.6 `--reasoning off` precedent).
- `--chat-template-file /config/chat_template.jinja` pointing at the fetched froggeric template.

### VRAM / HAMi budget

The MTP chart's v1.0.0→v1.0.1 fix (raising `CUDA_DEVICE_MEMORY_LIMIT_0` from `22000m` to `24000m` after a documented on-device OOM inside `common_speculative_init_result`) is the single most important cautionary precedent here. A DFlash2 drafter adds its own KV cache and compute buffers on top of the target model — do the arithmetic explicitly in the PR body:

- Target model (UD-Q4_K_XL) + q8_0/q8_0 KV @ 131K context ≈ same ~22.3 GB baseline documented in the MTP chart's v1.0.1 commit message.
- Add the DFlash2 drafter's own weights (~1.1 GB) plus its KV cache/compute buffer overhead (unknown until measured — this is exactly the class of allocation that OOM'd the MTP chart).
- Start `CUDA_DEVICE_MEMORY_LIMIT_0` at `24000m` (the MTP chart's already-fixed value) as a floor, not a target — expect you may need to reduce context (e.g. 96K) or drop to `Qwen3.8-27B-UD-Q3_K_XL.gguf` (13.1 GB fallback, already named in PR #19's own summary) if the drafter pushes you over budget on the 24 GiB 5090M.

### Chart file structure (mirror PR #19 exactly, renamed)

```
llamacppqwen38dflashone/
├── Chart.yaml                     # name: llamacppqwen38dflashone, version 1.0.0
├── OlaresManifest.yaml            # appid: llamacppqwen38dflashone, apiVersion v2, manifest v0.10.0
├── values.yaml                    # copy verbatim from llamacppqwen38mtpone/values.yaml (unchanged pattern)
├── owners                         # copy verbatim (aamsellem)
├── .helmignore                    # copy verbatim
├── templates/
│   └── deployment.yaml            # ConfigMap + Deployment + Service, --- separated (see below)
└── i18n/
    └── en-US/
        └── OlaresManifest.yaml    # en-US title/description override, mirror PR #19's i18n file
```

Fetch the exact current contents of `llamacppqwen38mtpone/Chart.yaml`, `OlaresManifest.yaml`, `templates/deployment.yaml`, `values.yaml`, `owners`, `.helmignore`, and `i18n/en-US/OlaresManifest.yaml` from branch `feat/add-qwen38-27b-mtp` in this repo (or from `main` once merged) and use them as your literal starting templates — do not write these files from scratch. Apply the deltas below:

**Chart.yaml**: rename `name` to `llamacppqwen38dflashone`; `version`/`appVersion` start at `'1.0.0'`; update `description` to reference DFlash2 speculative decoding instead of MTP.

**OlaresManifest.yaml**: rename `metadata.name`, `metadata.appid`, `entrances[0].name/host`, icon URL path, to `llamacppqwen38dflashone`. Update `title` (max 30 chars, `[a-z0-9A-Z-\s]` only — e.g. "Qwen38 27B DFlash" fits). Update `bento.badge` to something like `dflash2 · upstream` (or `dflash2 · fork` if you had to fall back to buun). Set `bento.hero.value` to `"TBD t/s"` — **do not invent a number**; PR #19's own precedent (and the repo's convention per recently merged PR #17) is to only publish measured numbers, and this PR explicitly must not fabricate a benchmark. Set `bento.hero.label` to something like `"DFlash2 n=7 · 131K"`. Update `bento.capabilities.mtp` to `false`/remove, and consider adding a `dflash: { enabled: true, accept: null }` capability if the bento schema in use elsewhere supports arbitrary capability keys (check other charts' manifests, e.g. `llamacppqwen36a3bdflashone`, for the actual schema key used for DFlash before inventing a new one). `upgradeDescription` and `fullDescription` must describe DFlash2, the drafter model/file used, the backend decision (upstream vs fork) and why, and must state plainly that on-device numbers are pending.

**templates/deployment.yaml**: base the ConfigMap on `llamacppqwen38mtpone`'s (same `TARGET_MODEL`/`TARGET_FILE`/chat-template fetch env vars), add `DRAFT_REPO`/`DRAFT_FILE` env vars matching the pattern in `llamacppqwen36a3bdflashone`'s ConfigMap. In the Deployment, keep the same `fix-shared-perms` alpine init container, the same shared-model hostPath + migration logic, the same `/config` emptyDir + chat-template fetch logic from the MTP chart, and add a second `curl` download block for the DFlash2 drafter GGUF (model this on `llamacppqwen36a3bdflashone`'s `DRAFT_FILE`/`DRAFT_REPO` download block — same `.ok`-marker idempotency pattern). Rename all k8s object names (`ConfigMap` name, `Deployment` name/labels, `Service` name/labels, container name) to something containing `llamacppqwen38dflashone` (or `llamacpp-qwen38-dflash-*`) — do not collide with the MTP chart's object names. Add `--model-draft "$SHARED_DIR/$DRAFT_FILE"` and the `--spec-type draft-dflash --spec-draft-n-max ...` flags to the `llama-server` exec line in place of the MTP chart's `--spec-type draft-mtp` flag.

**i18n/en-US/OlaresManifest.yaml**: mirror PR #19's i18n override file, updated for the new title/description.

### Validation checklist before opening the PR (mirror PR #19's own checklist)

- [ ] `helm lint llamacppqwen38dflashone/` passes
- [ ] `helm template llamacppqwen38dflashone/` renders clean (ConfigMap + Deployment + Service)
- [ ] `helm package` produces a valid `.tgz`
- [ ] `npm run build:catalog` — new chart appears in `src/catalog.json` with a generated 8-char hex `appid`, and the `.tgz` is embedded in `src/charts.json`
- [ ] Explicitly state in the PR: on-device install/benchmark on `192.168.1.32` has NOT been done yet, matching PR #19's own "Honest note: no on-device numbers yet" section. Do not put any t/s number into the manifest or PR body that has not actually been measured on Olares One hardware.
- [ ] Document the exact on-device validation sequence (mirror PR #19's `kubectl logs`/`curl localhost:8000/health`/generation-sweep steps), adapted for verifying DFlash2 draft acceptance instead of MTP acceptance.

### PR body requirements

- Explicitly reference PR #19 and the community comment that motivated this chart (bayerhazard's DFlash2 config report).
- State the backend decision (upstream `--spec-type draft-dflash` vs `buun-llama-cpp` fork) and the evidence for that decision (did you confirm the flag/drafter compatibility, or did you have to fall back — and why).
- Include the VRAM/HAMi budget arithmetic above, and state the starting `CUDA_DEVICE_MEMORY_LIMIT_0` value with justification tied to the MTP chart's documented OOM history.
- Do not include invented performance numbers. Use `TBD t/s` exactly as PR #19 did, and commit to updating it after on-device validation.
- Link the chart as a sibling to `llamacppqwen38mtpone`, explicitly stating both charts are intended to coexist (stable/upstream MTP tier vs fast DFlash2 tier), following the existing `llamacppqwen36mtpone` / `llamacppqwen36a3bdflashone` precedent in this same repo.
