# Novelty Check Report — Hierarchical Modality Credit Attribution for Multimodal Agent Memory

**Date**: 2026-05-11
**Stakes**: Advisor sync 2026-05-18 (7 days). Commit-or-pivot decision required.

---

## Verdict: **RISKY** — 必须 reposition，但不要 pivot 到 backup

核心数学 **就是 Owen value (1977)**。如果以"new hierarchical Shapley theory"卖论文，会被 reviewer 一击毙命。但若 reposition 为 **systems/empirical contribution**（首次把 Owen-value 归因落到 multimodal agent memory，配合 modality-pruned retrieval 实证），还能撑起 top-venue 论文 —— **前提是 Claim 2 实证 ≥10pp ΔAcc 必须真实兑现**。

---

## Core Claims × Novelty Status

| # | Claim | Status | Closest prior |
|---|---|---|---|
| 1 | Hierarchical / nested / two-level Shapley applied to any memory or RAG | **DONE in theory; NOVEL in this application** | **Owen 1977** (classical), Kamijo 2009 (two-step formulation), Hierarchical Contrastive Shapley (2512.19363, data valuation) |
| 2 | Multimodal attribution for memory-equipped agents | **NOVEL** (first application) | M3-Agent / WorldMM / MemVerse 不做归因；MM-SHAP / MultiSHAP 单模单步 |
| 3 | Per-memory-item modality decomposition | **NOVEL** (specific to memory hierarchy) | MM-SHAP (flat modality, no memory), Dr. SHAP-AV (2 modality, AVSR) |
| 4 | Item × modality two-level decomposition (any setting) | **DONE = Owen value** | Owen 1977, Two-Step Shapley (Kamijo 2009) |
| 5 | Variance reduction theorem (Σ 2^{L_k}) / 2^{Σ L_k} | **TECHNICALLY NOVEL but trivial** | Owen-MC variance bounds 一般化已有；这个具体 ratio 没人写过，但 reviewer 会说 "trivial once you write two games" |
| 6 | Modality-pruned retrieval | **NOVEL** | 现有 multimodal RAG 不做 attribution-driven modality pruning |

**结论**: 6 个 claim 里 2 个 (claim 1, 4) 在数学上被 Owen value 覆盖；4 个在 application 层面真正新；1 个 (claim 5) 技术上新但 reviewer 看来是 Owen-value 平凡推论。

---

## Closest Prior Work Matrix

| Paper | Year | Venue | 重叠维度 | 与我们的关键差异 |
|---|---|---|---|---|
| **Owen 1977** (classical) | 1977 | game theory | **two-level factorization 本身** | Owen 假设 fixed coalition structure；我们的玩家集 query-conditioned 动态；Owen 不涉及 ML/memory 应用 |
| **Kamijo 2009** (Two-Step Shapley) | 2009 | game theory | 显式 two-step computation | 同上，纯理论 |
| Hierarchical Contrastive Shapley (2512.19363) | 2025 | arxiv | hierarchical Shapley + Õ(1/√T) bound | 不同 domain（data valuation），不涉及 modality / memory；不引 Owen |
| **MM-SHAP** (Parcalabescu 2024) | 2024 | EACL | modality-level Shapley | flat 只到 modality 层；single-shot VL 分类；无 memory；无 hierarchy |
| **MultiSHAP** (2508.00576) | 2025 | arxiv | Shapley Interaction Index for VL | flat patch×token 二元交互；"two-level" 指 instance-vs-dataset，不是 item×modality；single-shot |
| **Dr. SHAP-AV** (2603.12046) | 2026 | arxiv | Shapley for AVSR | 仅 audio vs vision 二模；single utterance；无 memory hierarchy |
| **AgentSHAP** (2512.12597) | 2025 | arxiv | Shapley for LLM agent tool importance | tools as players，不是 memory items；text-only |
| **Audio LLM Modality Contribution** (2509.20641) | 2025 | arxiv | 用 MM-SHAP | flat，single-shot，音乐领域 |
| **Source Attribution in RAG** (2507.04480) | 2025 | arxiv | per-document Kernel SHAP | 单模 RAG; document atomic player |
| **TokenShapley** (2507.05261) | 2025 | ACL findings | token-level Shapley | flat; 玩家粒度过细; 单模 |
| **Circuit Analysis of Agent Memory** (2605.03354) | 2026 | arxiv | text agent memory diagnosis | 不用 Shapley；text-only；mechanistic interp 角度 |
| **M3-Agent** (2508.09736) | 2025 | open code | memory架构 + RL | 不做 attribution；codebase / benchmark donor |
| **WorldMM** (2512.02425) | 2025 | arxiv | multimodal memory + retrieval | 同上 |
| **MMA** (2602.16493) | 2026 | arxiv | multimodal memory | 同上 |

---

## Brutal honesty: 真正剩下的 novelty

把所有 Owen-classical 部分剥离后还剩什么：

### Genuine novelty (能挂在 paper 上)

1. **First application of Owen-value-style hierarchical attribution to multimodal memory-equipped agent** — 没人这么干过；application 级别的差异化 clear
2. **Variance ratio (Σ 2^{L_k}) / 2^{Σ L_k} 给出一个 sample-allocation heuristic** —— outer 采几个、inner 采几个，可以推出 closed-form 最优分配。这是真正可写的算法贡献
3. **Modality-pruned retrieval 算法 + 实证** —— attribution-driven 的 modality-level eviction，作为可执行 corollary。如果 ≥10pp 兑现，是 must-accept hook
4. **Live-agent-loop 速度可达性** —— hierarchical 的 O(K + Σ L_k) 采样让 attribution 能在 inference time 跑（flat MM-SHAP 在 K=4, L_k=4 的 multimodal memory 上算不动）

### 不是 novelty（不要在 paper 里 oversell）

- Two-level factorization 形式（= Owen 1977）
- Conservation theorem at each level（= Owen Efficiency axiom）
- Uniqueness on hierarchical lattice（Prop 3）（= Owen characterization theorem）
- 一般 McDiarmid concentration 应用（= 经典 Hoeffding-style 结果）

---

## Repositioning advice (5/18 之前必须做)

### Old framing (问题大)
> "Two-level factorized Shapley uniquely preserves Shapley axioms on the hierarchical lattice with finite-sample variance reduction theorem"
- Reviewer Round 1 评语会是: "This is Owen value. Cite Owen 1977. Reject as no new theory."

### New framing (RISKY 但有救)
> "We bring Owen-value attribution to the **memory layer of multimodal agents** for the first time. Aligned with the natural item → modality patch hierarchy, it enables **fast enough attribution to run inside a live agent loop**, yields a **closed-form sample-allocation policy** for outer-vs-inner Monte Carlo draws, and powers **modality-pruned retrieval** that delivers ≥10 pp accuracy lift on M3-Bench."

### 具体修改清单

1. **Abstract / Introduction**:
   - 删 "novel two-level Shapley factorization"
   - 换成 "first systematic application of Owen-value attribution to memory-equipped multimodal agents"
   - Lead with empirical promise: M3-Bench 上 modality-pruned retrieval 的 accuracy lift

2. **Theory section (main text)**:
   - Prop 1 (conservation) → 1 句话或脚注，引 Owen 1977
   - Prop 3 (uniqueness) → 完全移到 appendix，标题 "Background: Owen Value Characterization"
   - **保留 main text 的核心 theoretical hook**:
     - Variance ratio bound (Corollary 2c) 作为 "sample-efficiency 分析" 而不是 "新定理"
     - 由此导出 **Optimal Outer-vs-Inner Sample Allocation Policy** —— 这是 derived but possibly publishable
   - 加一个 "Related: Owen value" 小节诚实承认 classical 渊源

3. **Method**:
   - 把 "新算法" 框架改为 "Owen-value computation + 4 个工程级优化（Kernel SHAP outer + MC permutation inner + top-N=2 modality + attention-logit seeding + audio super-patch conditional aggregation）"
   - 把 modality-pruned retrieval 升级为 supporting algorithm 的 first-class contribution，不再是 "demotable"

4. **Validation**:
   - Claim 1 (detection)、Claim 1.5 (synthetic scaling)、Claim 1.6 (LOO variance) 全部保留 —— 现在是 method-level evidence
   - **Claim 2 必须达到 ≥10pp ΔAcc** —— 不能 demote 了；这是 paper 的存活条件
   - 加 Claim 3: **cross-architecture generalization** —— 在 ≥3 个不同 memory agent 架构（M3-Agent / WorldMM / MMA）上跑同一套 attribution，证明方法不绑特定 backbone

5. **Target venue 调整**:
   - 老定位（theory paper）: NeurIPS / ICML
   - 新定位（systems/applied）: **NeurIPS Datasets & Benchmarks** track 或 **CVPR 2026** application track 更合适；ICLR 也行；纯 theory-heavy 的 venue（COLT / ALT）已不可能

---

## Decision: Commit 还是 Pivot

**Recommendation: COMMIT to the RISKY repositioning. 不 pivot 到 backup ②。**

理由:
1. Backup ② (Cross-Modal Causal Mediation for Write-Policy) 的 novelty 风险**同样存在** —— CMA 在 multimodal classification 已有论文 (Unveiling Modality Bias 2511.05883, Disentangling Bias via CMA 2508.04999)；pivot 不解决问题
2. 当前方向 application novelty (memory-equipped agent) + empirical bar (≥10pp on M3-Bench) 还**足以撑住 top venue**，前提是实证兑现
3. 7 天 timeline 紧，pivot 重启代价大；reposition 只需要重写 abstract/intro/method framing，~1 天工作量
4. 师兄之前明确说"modality credit"主线 > "agentic memory tool-call"备选 —— pivot 到 ② 在方向上更偏离师兄推的主线

**风险敞口**:
- 如果 Claim 2 实证 < 10pp ΔAcc，paper 没有 must-accept hook；这种情况下退路是 demote 到 NeurIPS / ICLR **workshop**（如 ICLR 2026 MemAgents workshop，2/13 deadline 已过；NeurIPS 2026 Memory workshop 待开放），或转 **systems venue**（MLSys, EuroSys）

---

## 5/18 sync 必须跟师兄确认的 3 件事

1. **Owen value 重叠问题**: 师兄知不知道 Owen 1977？是否 OK 我们 reposition 成 "first application"？
2. **Stretch goal 门槛**: ≥ 10pp 是否合理？还是接受 ≥ 5pp + 显著 context reduction (≤20%) 也算成功？
3. **Venue 调整**: 师兄是不是仍然认 NeurIPS / ICLR 这种 theory-friendly venue？或者愿意走 systems / applied track？

---

## Suggested Positioning Statement (for advisor sync 1-pager)

> 把 **Owen value (1977)** 这一经典 cooperative game theory 工具，**首次** 落到 multimodal memory-equipped agent 的归因问题上。核心 contribution 三层：
>
> 1. **Application novelty**: 把 (memory item × modality patch) 的自然嵌套结构与 Owen value 的 coalition structure 对齐 —— 让 attribution 既保 conservation 又能在 live agent loop 里 inference-time 跑通
> 2. **Algorithmic contribution**: 从 variance ratio 推出 closed-form outer-vs-inner sample allocation policy，让 hierarchical attribution 在 K=8, L_k=6 regime 比 flat MM-SHAP 快 ~10×
> 3. **Empirical contribution**: 在 M3-Bench / WorldMM / VideoWebArena 上做 modality-credit detection 实验 + modality-pruned retrieval 实证 (target ≥10pp ΔAcc, ≤40% context retention)，并在 ≥3 个 memory agent 架构上验证 generalization

---

## Sources

- [Owen 1977 — Values of games with a priori unions](https://www.sciencedirect.com/science/article/pii/0167404877900028) (need to verify citation)
- [Kamijo 2009 — Two-Step Shapley for Coalition Structures](https://www.worldscientific.com/doi/abs/10.1142/S0219198909002261)
- [Hierarchical Contrastive Shapley Values (2512.19363)](https://arxiv.org/html/2512.19363)
- [MultiSHAP (2508.00576)](https://arxiv.org/abs/2508.00576)
- [MM-SHAP (2212.08158)](https://arxiv.org/html/2212.08158)
- [Dr. SHAP-AV (2603.12046)](https://arxiv.org/abs/2603.12046)
- [AgentSHAP (2512.12597)](https://arxiv.org/pdf/2512.12597)
- [Audio LLM Modality Contribution (2509.20641)](https://arxiv.org/html/2509.20641)
- [Circuit Analysis of Agent Memory (2605.03354)](https://arxiv.org/html/2605.03354v2)
- [Towards Attributions of Input Variables in a Coalition (2309.13411)](https://openreview.net/forum?id=h5TXCnnEyy) — coalition-aware attribution framework, worth deep-read
