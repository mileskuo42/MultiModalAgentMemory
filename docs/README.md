# Context Documents

This folder collects every design document, novelty check, and proposal version
that informs the codebase. **Read in this order if you are an implementation
agent picking up the project**:

| Order | File | What it contains | When to read |
|---|---|---|---|
| 1 | [`proposal_brief.md`](proposal_brief.md) | 5-minute overview of what we're doing and why. Start here. | First thing |
| 2 | [`experimental_framework.md`](experimental_framework.md) | The full implementation guide — Phase-by-Phase what to run, what to measure, decision gates. Cross-referenced with the code skeleton in `../modality_credit/`. | Before writing code |
| 3 | [`method_plan_repositioned.md`](method_plan_repositioned.md) | Current (v2) method plan. Score 8.5/10. Has Owen-value reposition, closed-form allocation policy, modality-pruned retrieval as must-accept hook. **This is the source of truth for method.** | When implementing |
| 4 | [`novelty_report.md`](novelty_report.md) | Deep novelty-check findings (2026-05-11). Identifies Owen 1977 as the classical foundation; documents the 5+ closest 2025–2026 neighbors and how we differ. **Read before any related-work writing.** | When writing paper |
| 5 | [`idea_report.md`](idea_report.md) | Upstream idea-generation output. 10 candidate directions, devil's advocate, final pick. Mostly historical now. | For context only |
| 6 | [`method_plan_v1.md`](method_plan_v1.md) | Original method plan, pre-novelty-check. Kept for diff. **Do not use as source of truth** — superseded by v2. | Diff reference |
| 7 | [`proposal.html`](proposal.html) | Pretty rendered version of the proposal (Claude design system). For sharing with humans. | For sharing |

## Project state (as of 2026-05-12)

- **Stage**: pre-experiment, framework skeleton complete
- **Advisor sync**: 2026-05-18
- **Target venue**: NeurIPS 2026 D&B / CVPR 2026 application / ICLR 2027
- **Critical gates**:
  - Phase 1 (5/14): Toy pilot. If variance > 3× bound OR conservation residual > 10% → pivot to backup
  - Phase 3 (5/16): Claim 2 ΔAcc ≥ 10pp → top venue / 5–10pp → second tier / <5pp → workshop

## Three "must-not-break" invariants

1. **Permutation invariance is load-bearing.** Owen value requires it. If `audits.positional_invariance` reports residual > 10%, the whole framework is in doubt.
2. **Conservation residual is the math sanity check.** If `audits.conservation_residual` reports mean > 10% after normalization (`u_norm`), the estimator implementation has a bug.
3. **`Utility.evaluate()` is the only expensive call.** Everything else must be O(K · L_k). Caching is mandatory.

## Backup direction

If main line fails Phase 1: **Cross-Modal Causal Mediation for Write-Policy**.
- Uses Pearl-style do-calculus, not Shapley/Owen.
- See `idea_report.md` §"Backup direction" for the full sketch.
