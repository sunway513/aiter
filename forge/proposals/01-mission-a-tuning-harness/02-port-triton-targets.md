# [forge][Mission A] Port 11 sub-project-2 Triton optimization targets from sunway513/aiter-forge

Refs RFC #TBD-RFC.

## Goal

Move the Triton-kernel optimization target backlog out of the soon-to-be-archived `sunway513/aiter-forge` repo and into ROCm/aiter as forge target definitions plus tracker issues.

## The 11 targets (originally `sub-project-2` label on `sunway513/aiter-forge`)

1. PA Decode (Triton)
2. PA Prefill (Triton)
3. Extend Attention (Triton)
4. RMSNorm (Triton)
5. MoE GEMM A8W8 BlockScale (Triton)
6. MoE BF16 (Triton)
7. GEMM A8W8 BlockScale (Triton)
8. GEMM AFP4WFP4 (Triton)
9. FF A16W16 Fused (Triton)
10. MLA Decode + RoPE (Triton)
11. RoPE (already partially seeded under `forge/targets/flydsl_rope/`)

## Scope

- For each target: create `forge/targets/<name>/target.yaml` skeleton with shape sweep, baseline command, success criterion.
- File one ROCm/aiter issue per target with label `area: forge`, `forge: tuning-target`, and the original sub-project-2 carry-over context.
- Close the corresponding sunway513/aiter-forge issue with a "transferred to ROCm/aiter#X" pointer.

## Out of scope

- Actually running optimizations on each target (each will be its own follow-up).
- FlyDSL or Opus targets — those continue under `forge/targets/flydsl_*` independently.

## Definition of done

- 11 target.yaml skeletons exist under `forge/targets/`.
- 11 issues filed against ROCm/aiter, all labeled and cross-referenced.
- 11 issues on sunway513/aiter-forge closed with redirect.

## Owner

forge maintainers + one Triton-kernel engineer.

## Estimated effort

M — mostly mechanical transfer + skeleton authoring.
