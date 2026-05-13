# Owen-Value Attribution for Multimodal Agent Memory
## 短报告 — 我们打算做什么，怎么做

**Date**: 2026-05-12 · **Advisor sync**: 2026-05-18 · **Target venue**: NeurIPS 2026 D&B / CVPR 2026 application / ICLR 2027

---

## 一、要做的事

第一次把 **Owen value (1977)**（cooperative game theory 里处理 "玩家天然分组" 的经典 hierarchical Shapley）落到 multimodal memory-equipped agent 的归因问题上。

具体场景：M3-Agent / WorldMM / MMA 这类 agent 把多模观察（视频帧 + caption + 音频）存进 long-term memory，提问时 retrieve K 条 episode 喂给 base VLM (Qwen-2.5-VL) 答题。当 agent 答错时，我们要精确指出 **"是第 k 条 memory 的哪种 modality (vision / text / audio) 在起作用"**——不仅做事后解释，还能拿归因结果直接 prune 掉低贡献 modality patch，把 accuracy 推高。

**核心 contribution 三层**:
1. **Application novelty**: 第一次把 Owen value 用到 multimodal memory agent。Deep novelty check (2026-05-11) 已确认 (Owen) × (item × modality) × (memory agent) × (inference-time) 四维交集无人做过。
2. **Algorithm**: 从 variance ratio bound 推出 closed-form sample-allocation 策略 N_outer* ∝ √(2^K), N_inner^(k)* ∝ √(2^{L_k})。给定预算 B 自动得到最优配置，无需调参。K=8/L_k=6 时把所需 utility query 从 16.7M 降到 272 次（≈ 61,679× 节省）。
3. **Empirical**: modality-pruned retrieval 在 M3-Bench / LongVideoBench / 跨 ≥3 architecture 上目标 ΔAcc ≥ 10pp, context retention ≤ 40%。

---

## 二、怎么做

### Method

```
Query Q
   ↓
M3-Agent retrieval (frozen)  →  M(Q) = {m_1, ..., m_K},  m_k = (v_k^vision, v_k^text, v_k^audio)
   ↓
Owen-Value Estimator (NEW, inference-time, 0 training):
   ├ Outer game:  Kernel SHAP over K items   → {φ_k}
   ├ Inner game:  MC permutation over L_k modalities → {ψ_{k,ℓ}}, top-N=2 items
   └ Sample budget split by Alg. 1
   ↓
Generator (frozen, Qwen-2.5-VL-7B)
   ↓
Modality-Pruned Retrieval (NEW):  M̃(Q) = {v_k^(ℓ) : φ_k > τ_1 ∧ ψ_{k,ℓ} > τ_2}
```

### 为什么 Owen value 在这个 setting 特别合适

Retrieval 把 agent 的动态因果流切成**静态 snapshot 集合**——所有 K 条 retrieved item 并行 attend，permutation invariance 自然成立。不像 OSPO (RL credit) 被 trajectory 因果约束，我们可以自由 mask 任意 (item × modality) 子集来评估 utility。**memory 的真正时序在 write-time，我们主动切割那一层**（frozen retrieval, non-goal: write-policy）。

加分项：M3-Agent 的 memory 本来就是 entity-centric multimodal graph，**每个 entity 下的 modality 是 separate sub-nodes**。两层结构是 M3-Agent 已有的数据结构，我们不是 patch 不存在的 hierarchy，是激活已有 hierarchy。

### Theory (main text only Prop 2 + Alg. 1)

- **Prop 2**: Var(φ̂_k) ≤ 2^K/(4N), McDiarmid concentration P(|φ̂_k − φ_k| > ε) ≤ 2 exp(−2Nε²/2^K)
- **Corollary 2c**: Var(hierarchical) / Var(flat) ≤ (Σ 2^{L_k}) / 2^{Σ L_k}
- **Alg. 1**: 从 2c 拉格朗日最优化推出 N_outer* / N_inner^(k)*
- **Phase 4 加 matching lower bound** → "tight finite-sample bound"
- Prop 1 (Conservation) 和 Prop 3 (Uniqueness) 都是 Owen 1977 直接推论 → 移到 Appendix A "Background: Owen Value"，不卖成新 theory

### 实现

~200 LoC patch 到 `m3_agent/retrieval.py`，0 training, frozen base model and retrieval。

---

## 三、实验

### Phase 1–3 (5/13–5/16, 7.1 GPU-h, 主线)

| Phase | Day | 内容 | 目标 / 决策点 |
|---|---|---|---|
| 1 | 5/14 | Toy pilot, 30 QA, K=4 | variance 收敛 + conservation < 5% → 否则 pivot to backup |
| 2 | 5/15 | Claim 1 (100-QA detection) + 1.5 (synthetic 验证 Alg. 1) + 1.6 (real-data LOO) | ΔAcc ≥ 15pp, 理论与实证 ±20% 一致 |
| 3 | 5/16 | Claim 2 (**modality-pruned retrieval, must-accept hook**) + Claim 3 (跨 3 架构 M3-Agent/WorldMM/MMA) | **≥10pp ΔAcc 决定 venue tier**, 跨架构 ±3pp |

### Phase 4 (5/19–6/03, +13 GPU-h, hardening pack)

| Pillar | 内容 | 加分 |
|---|---|---|
| A | Memory poisoning detection (4 类 adversarial injection, AUROC ≥ 0.85) | ⭐⭐⭐⭐⭐ |
| B | Scale to LongVideoBench K=8 (300 QA), 把 sample-complexity wedge 从 table 变 figure | ⭐⭐⭐⭐ |
| C | Matching lower bound 理论补强 (Prop 2 升级为 tight bound) | ⭐⭐⭐⭐ |
| D | Failure mode taxonomy (re-cluster Phase 2 失败 case by (φ, ψ) 分布) | ⭐⭐ |
| E | Human study (20 annotators, diagnosis accuracy + time, with vs without attribution) | ⭐⭐⭐⭐⭐ |

主线 + Phase 4 ≈ 20 GPU-h, ~3 周。Top-venue 概率从 ~25% 推到 ~65–70%。

---

## 四、风险与备选

| 触发 | 应对 |
|---|---|
| Phase 1 variance > 3× bound | 立即 pivot to backup direction: **Cross-Modal Causal Mediation for Write-Policy** (Pearl-style do-calculus，工具切换，方向不偏) |
| Claim 2 ΔAcc < 5pp | Demote 主要 empirical hook，paper 仍可靠 Claim 1+1.5+1.6 + Phase 4 投 ICLR workshop |
| Claim 3 跨架构 fail (只在 M3-Agent work) | Reframe 为 "M3-Agent case study"，venue tier 掉一档 |

---

## 五、Advisor sync 5/18 必问

1. **Owen 1977 重叠** — 接受 "first application of Owen value to multimodal memory agent" 这一 reposition，而不是 "new theory"?
2. **Empirical bar** — Claim 2 ≥ 10pp 合理 OR 改 ≥ 5pp + 20% context reduction 也算成功？
3. **Venue path** — 走 NeurIPS D&B / CVPR application / ICLR 而非 theory-heavy ICML/NeurIPS main，OK?
4. **Alg. 1 closed-form sample allocation** — 算 second main contribution 吗？还是嫌 too applied？
5. **Phase 4 hardening pack** — 全套 (A+B+C+D+E) buy-in 吗？或者哪几个不必要？

---

**相关文档**:
- 详细 method plan: `multimodal_memory_credit_method_plan_repositioned.md` (~500 行)
- Novelty check: `multimodal_memory_credit_novelty_report.md`
- Idea report (上游): `multimodal_memory_credit_idea_report.md`
