# Hypothesis — aiter_flydsl_fmha_gfx1201_cleanup

## Claim

Skill-guided cleanup of the gfx1201 FlyDSL FMHA kernel — replacing raw MLIR
dialect ops (`arith.*`, `scf.*`, `vector.*`, `memref.*`) with FlyDSL internal
types (`fx.Int32`, `fx.Index`, `Vec`, `range(..., init=[...])`, etc.) per
`.claude/skills/flydsl-internal-types-cleanup/SKILL.md` — preserves the kernel
along three axes:

1. **Final ISA byte equality.** `sha256sum ~/.flydsl/debug/flash_attn_func_gfx1201/21_final_isa.s` is identical between pre-cleanup and post-cleanup builds for cleanup groups G1 (constants), G2 (index casts), and G3 (vectors). Strict equality is the success criterion for these groups.
2. **Correctness.** Cosine similarity vs PyTorch SDPA on the Wan2.1 production shape (B=1, S=32768, H=12, D=128, bf16, non-causal) is ≥ 0.999985 — at least as good as the pre-cleanup baseline.
3. **Performance parity.** Per-shape kernel TFLOPS stay within ±2% of pre-cleanup. End-to-end Wan2.1 wall time stays within ±2% of the 460.9 s pre-cleanup baseline (i.e. ≤ 466 s).

For G4 (truncations + Python-operator arithmetic) and G5 (control-flow rewrite
to `@flyc.jit` dispatch + `range(init=[...])` loops), the ASM hash is *expected*
to drift because the lowering path differs even when the MLIR is semantically
equivalent. For these groups the success criterion downgrades to perf parity
(±2%) and cosine sim ≥ 0.999985, with no ISA equality requirement.

## Falsifiability

The claim fails — and the relevant cleanup group must be reverted — if any of
the following holds after applying the group:

- G1/G2/G3: `sha256sum 21_final_isa.s` differs from baseline AND per-shape
  kernel TFLOPS regress by more than 2%.
- Any group: cosine sim vs SDPA on the Wan2.1 shape drops below 0.999985.
- Any group: e2e Wan2.1 wall time exceeds 466 s (1.011× baseline).
- Any group: kernel fails to compile under `FLIR_CHIP=gfx1201`.

Each of these is a hard, measurable signal — none requires interpretation.

## Why this should hold

The skill's design intent is "syntactic sugar over the same MLIR ops": every
preferred form lowers to the same `arith` / `vector` op the explicit form
uses. The "Important Exceptions" list (preserved verbatim in
`replacement_map.json` under `exceptions_keep_lowlevel`) catches every case
where the lowering would diverge — fastmath flags, unsigned division,
volatile/nontemporal loads, ROCDL intrinsics, MFMA/WMMA. The cleanup loop is
group-by-group with ASM hash checking precisely so we catch any unexpected
divergence the moment it appears.

If even G1 (which is a pure constructor swap — `arith.constant(0, type=T.i32)`
→ `fx.Int32(0)`) fails ASM equality, the skill's premise is wrong and we
should escalate upstream to FlyDSL before continuing.

## Out-of-scope

This target does NOT attempt to improve perf. Any speedup discovered during
cleanup is logged as a learning entry but the kernel ships at parity. Perf
tuning belongs in `targets/flydsl_fmha_gfx1201`, not here.
