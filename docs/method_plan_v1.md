# Hierarchical Modality Credit Attribution for Multimodal Agent Memory

**Status**: Pre-experiment method plan, 4 rounds of GPT-5.4 (o3 xhigh) review.
**Score**: 8.5/10 (proposal-stage ceiling ~8.7; path to 9+ requires Claim 2 empirical success).
**Verdict**: Ready for advisor sync on 2026-05-18. Implementation-ready in ~200 LoC.
**Backup if pilot fails**: Cross-Modal Causal Mediation for Write-Policy (idea ② in idea report).

---

## TL;DR (elevator pitch)

> 想象一个 multimodal agent，能看视频、听音频、读字幕，把这些当 episode 存进 long-term memory。你问它一个问题，它从 memory 里 retrieve 几条相关 episode 来回答。**问题**：它答错的时候你完全不知道是哪条 memory 害的——更别说是这条 memory 里的画面、字幕、还是音频出的问题。现有归因工具要么只懂单模态文本检索，要么把整条 multimodal memory 当一个原子玩家，把 item 内部的模态结构整个抹掉。
>
> **我们的 idea**：既然 memory 天然是两层嵌套——K 条 item，每条 item 内部 L_k 个 modality——那归因也按这两层来分摊。外层 game 决定哪条 memory 贡献大，内层 game 在 item 内部分摊哪个 modality 贡献大。
>
> **为什么好**：(1) 0 训练，纯 inference-time，~200 行代码 patch 到 M3-Agent；(2) 嵌套结构本身就是 sample complexity 的 wedge——flat token Shapley 要查 65,536 次，hierarchical 只要 80 次，800× 便宜，而且我们能证明 finite-sample variance ratio 严格 bounded（这是核心 theorem）。
>
> **落地**：检测——告诉用户"agent 答错是因为它过度依赖了第 3 条 memory 的音频转录"；改进——把低贡献 modality patch 直接 prune 掉，省 context 且不掉分。检测→改进一气呵成。
>
> **一句话总结**：把多模 agent memory 的归因建模成 (item × modality) 两层 cooperative game，几乎免费拿到 800× 采样效率提升和可证明的 variance bound。

---

## Motivation: 为什么这是一个 well-defined 且未被解决的痛点

### 场景

2025–2026 涌现了一批 "memory-equipped multimodal agent"——M3-Agent (ByteDance)、WorldMM、MMA、MemVerse 这类系统让 agent 在长期交互中持续看视频、听音频、读对话，把这些观察以 episode 的形式存进 long-term memory。等用户提问，它从 memory 里 retrieve 出 K 条相关 episode，喂给 base VLM (Qwen-2.5-VL / InternVL) 来生成答案。

每条 episode 不是单一信号——它是一个**多模 composition**：比如 "厨房场景"这条 memory 包含 (视频帧 + caption '有人在切番茄' + 音频 '刀切菜板的声音' + 可选场景元数据)。一条 memory 至少 3–5 个 modality patch，K 通常是 4–8 条。

### 痛点

当 agent 答错的时候，开发者/用户/审计者三方都想知道一件事：**这次失败的 credit 应该挂在谁头上？**

- 是 retrieval 错了——拉出了不相关的 memory？
- 还是 retrieval 对了但 memory 里某个 modality 误导了 generator——比如 caption 写错了导致 audio 信号被覆盖？
- 还是某条 memory 的 audio 转录有 hallucination 把 generator 带跑偏？

这三个失败模式需要完全不同的修复策略，但当前 agent 的归因能力是**全瞎的**。最常见的做法是 black-box masking——挨条去掉 memory 看 accuracy 怎么变。但这只能告诉你 "去掉第 3 条 memory 准确率掉 4 pp"，回答不了 **"是第 3 条 memory 里的哪个 modality 起了关键作用"**。

而这恰恰是修 agent 时最有用的信息：知道是 audio 转录的问题，就去优化 ASR；知道是 caption 的问题，就去优化 captioner；知道是 vision frame 的问题，就去优化关键帧采样。

### 为什么这事现在才值得做

memory-equipped multimodal agent 在 2024 之前还是工程稀有品种；2025 之后才有可用的开源 codebase（M3-Agent）和 benchmark（M3-Bench / WorldMM eval / LoCoMo）。归因社区的工具链（Shapley / influence function / activation patching）此前主要服务于 single-shot classification 或 text-only RAG，没人把它落到这个 niche。窗口正好。

---

## Prior work landscape：5 条路径，每一条都差一点

把现有归因方法按 "玩家是什么" 拆成 5 类。理解每一类为什么不适配，就能看清我们的空白。

### Path 1: Per-document Shapley for RAG（最接近，但是单模 + atomic player）

**代表工作**: Source Attribution in RAG (arxiv 2507.04480, 2025)、ContextCite、MaxShapley (2512.05958)、Fair Document Valuation in LLM Summaries (2505.23842)、RepoShapley (2601.03378)

**它们做什么**: 给 RAG 系统的每条 retrieved document 一个 Shapley credit score。玩家 = 一条 document，masking = "include this document yes/no"。用 Kernel SHAP / surrogate model 快速估计。

**为什么对 multimodal agent memory 不够用——"atomic player" 是核心问题**:

打个比方。想象你订了一个 meal-kit subscription（每个 box 里有蔬菜 + 酱料 + 食谱）。Per-document Shapley 能告诉你 "第 3 个 box 对你这周的烹饪评分贡献最大"——但它没法告诉你 **"第 3 个 box 里的食谱有用，但里面的蔬菜其实是坏的"**。因为它把整个 box 当成一个不可分割的原子玩家——你只能 include 或 exclude 整个 box，没法做"保留食谱但拿掉蔬菜"这种 partial intervention。

放到 multimodal memory：一条 episode = "厨房场景 (视频帧 + caption + 音频)" 是 box；vision/text/audio 是 box 里的成分。Per-document Shapley 只会给你**整条 episode** 的一个分数。当你想问 "audio 转录写错了导致答错" vs "视频帧不够清楚" 时，这个分数完全帮不上忙——它把 modality 维度抹平了。

文本 RAG 没这个问题，因为一条 document 就是一坨同质文本，没有内部的模态结构需要再分。但对多模 memory 这就是结构性失配。

### Path 2: Token-level Shapley（玩家粒度太细，结构爆炸）

**代表工作**: TokenShapley (arxiv 2507.05261, 2025)

**它们做什么**: 把粒度推到极致——每个 token 是一个玩家。理论上能告诉你"是 caption 里的'切番茄'这三个 token 还是 audio 转录里的'刀切声'起了作用"。

**为什么不够用**: (a) **玩家数爆炸**。一条 multimodal episode 的 vision frame 经过 patch embedding 后常 ≥ 50 token；K=4 条就 ≥ 200 token 玩家。Shapley 需要 2 的玩家数次方个子集，2 的 200 次方根本算不动。即使用 Monte Carlo 估计，variance 也因为 token-level correlation 高得吓人——同一帧里相邻 patch 不是 i.i.d. 的。(b) **结构信息丢失**。"哪个 token 重要" 远不如 "哪个 modality 重要" 对修 agent 有用——修 ASR 模块比修 ASR 里第 17 个 token 容易得多。

我们的做法本质上是**找到一个中间粒度**：粗于 token，细于 atomic episode——正好落在 "item × modality" 这个天然嵌套的二层结构上。

### Path 3: Attention-rollout / Integrated Gradients / Saliency map（无 conservation，难解释）

**代表工作**: MMEL (multi-modal explainable learning), Cross-Modal Attention Guided Unlearning (CAGUL, 2510.07567), Circuit Tracing in VLMs (arxiv 2602.20330)

**它们做什么**: 用 attention weight 或 gradient 生成一张 saliency heatmap——画面/文本里哪些 token 被"看得多"。

**为什么不够用**: (a) **没有 conservation property**。Saliency 给你的是相对热度，加起来不一定等于总贡献，没法说"audio 贡献了 32% 的 credit"。(b) **不处理 dynamic retrieval pool**——saliency 是单次 forward 上的内部信号，跨多步 agent 交互不收敛。(c) **被反复批评的 spurious correlation**：高 attention 不等于因果重要，已有 NeurIPS 多篇论证。

Shapley 框架的优势恰好是 conservation（守恒）和因果解释（masking 是 counterfactual）。我们要保留这两个性质，但解决 atomic-player 问题。

### Path 4: Memory architecture papers（根本不做归因）

**代表工作**: MMA (Multimodal Memory Agent, 2602.16493——师兄今天分享的那篇), M3-Agent (2508.09736), WorldMM (2512.02425), MemVerse, A-Mem (NeurIPS 2025), MemGPT, SimpleMem (2601.02553)

**它们做什么**: 设计**新的 memory storage / retrieval / consolidation 架构**——modality-specific layers, Zettelkasten linking, episodic + semantic 双层, hierarchical compression。

**为什么不够用**: 这一整条 line **完全不做 attribution**。他们关心的是"怎么存得更好、检索得更准、消耗 token 更少"，但对"为什么这次答对/答错"这种 explanatory question 一概没碰。

这条线给我们提供了 **codebase 和 benchmark**——M3-Agent 是开源的 retrieval API，M3-Bench / WorldMM eval 是现成的评测集。我们的工作正好和他们正交：我们不重做架构，我们给他们的架构加上 "解释器"。

### Path 5: Single-shot multimodal classification attribution（场景错位）

**代表工作**: Unveiling Modality Bias (2511.05883)——用 Shapley + causal mediation 给 misinformation detection 做 modality credit；Disentangling Bias via Intra/Inter-modal Causal Attention (2508.04999) 给 sentiment 做 CMA

**它们做什么**: 在 **single-shot multimodal classification**（一张图配一段文字 → 类别标签）上做 modality importance 归因。**他们的玩家也是 modality**，看起来跟我们很近。

**为什么不够用**: (a) **没有 memory hierarchy**——他们的输入是单一 (image, text) 对，不是 K 条 retrieved episode 的集合。(b) **没有 dynamic player set**——输入 modality 集合是固定的，不随 query 变化。(c) **没有 sample complexity 问题**——单 sample 上的 Shapley 玩家数 ≤ 3，2 的 3 次方枚举即可。这条线已经做到极致，但放不进 memory-equipped agent 的 setting。

我们和他们的差异是 **outer game**——把 modality 归因嵌进 retrieval 这一外层。

### 总结：5 维差异化矩阵

| Prior line | Single-shot only | Hierarchical | Multimodal item | Memory-equipped | Finite-sample bound | Optimality |
|---|---|---|---|---|---|---|
| 1. Per-doc Shapley (Source Attr in RAG, ContextCite) | ✓ | ✗ | ✗ atomic | ✗ | partial | ✗ |
| 2. Token-level (TokenShapley) | ✓ | ✗ | ✓ (overshoot) | ✗ | ✗ | ✗ |
| 3. Attention / IG (MMEL, CAGUL) | ✓ | ✗ | ✓ (no conservation) | ✗ | ✗ | ✗ |
| 4. Memory architectures (MMA, M3-Agent, WorldMM) | ✗ | — | ✓ | ✓ (no attr at all) | — | — |
| 5. Single-shot modality CMA (UMB, IMCA) | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |
| **Ours** | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ (appendix) |

每条 prior line 都恰好缺**至少两个维度**，且这些维度同时缺位是结构性的——不是"加点工程修一下"就能补齐的。

---

## Problem Anchor (Immutable)

- **Bottom-line problem**: 在 multimodal LLM agent 的长期 memory 系统中，每次 answer 的 credit 如何被分摊到 (memory item k, modality patch ℓ, time step t) 三元组上？当 agent 答对/答错 query 时，**哪条** memory 起关键作用，**该条 memory 内的哪种模态**（视觉帧 / 文本字幕 / 音频转录）真正承担贡献？
- **Must-solve bottleneck**: 现有 attribution 方法（per-document Shapley for RAG、attention rollout、IG、TracIn、Datamodels）都假设玩家集合静态、玩家无内部结构、且单模。Memory-equipped multimodal agent 同时违反三点：(a) M(Q) 是 query-conditioned；(b) 每个 item 是多模 composition；(c) memory item 跨时间累积。
- **Non-goals**: 不做新 architecture / 不做 RL training / 不做 modality alignment / 不做 mechanistic interp circuit-level / 不做新 benchmark.
- **Constraints**: ≤ 8 GPU-h pilot; 7 天 formalize（advisor sync 2026-05-18）; M3-Agent codebase; Qwen-2.5-VL-7B 级 base; NeurIPS 2026 / ICLR 2027 target.
- **Success condition**: (1) Hierarchical Shapley 正式定义 + finite-sample variance theorem 完整证明; (2) M3-Bench 100-QA top-vs-bottom quartile prune ΔAccuracy ≥ 15 pp; (3) modality-pruned retrieval 保留 ≤ 40% patches 时 accuracy 下降 ≤ 2 pp.

## Technical Gap (桥接到 method)

把上面 5 条 prior line 的失配点合并起来：现成工具要么 (i) 玩家是 atomic episode 失去 modality 维度（Path 1），要么 (ii) 玩家是 token 失去结构 + 玩家数爆炸（Path 2），要么 (iii) 没有 conservation 难以可靠解释（Path 3），要么 (iv) 整条 line 根本不做归因（Path 4），要么 (v) 不处理 memory hierarchy（Path 5）。

Naive 补丁也不够：更大 context window 不解决归因；prompting "请告诉我你用了哪条 memory" 这种 self-report 已被多篇论文证明与 ground-truth attribution 弱相关；stack 一个新 attention module 只是引入更多需要被归因的参数。

**Smallest adequate intervention**：利用 memory 天然的嵌套结构定义 **two-level Shapley**——外层 K items，内层 item-k 的 L_k modality patches。嵌套结构本身就是 sample complexity 的 wedge：O(2^K + Σ 2^{L_k}) << O(2^{Σ L_k})。Time 维度通过 retrieval pool 在多步交互中的递归 attribution 自然嵌入，不用单独引入 time 玩家避免维度爆炸。

## Method Thesis

> 把 multimodal agent memory 的 credit attribution 形式化为 **two-level factorized Shapley over a query-conditioned dynamic player set with item-internal modality composition**，with **finite-sample variance reduction** relative to flat estimation (main quantitative claim).

- **Why smallest adequate**: 0 训练参数；inference-time estimator + threshold pruning；~200 LoC Python patch on M3-Agent
- **Why timely**: memory-equipped multimodal agent 2025–2026 才成熟（M3-Agent / WorldMM / MMA 都 ≥ 2025）；per-document RAG Shapley 饱和但单模；ICLR 2026 MemAgents workshop 已在推动 memory layer 标准化

## Contribution Focus

- **Dominant** (mainline): Two-level Shapley factorization as the correct attribution unit for multimodal agent memory, with **conservation** (Prop 1) + **finite-sample variance reduction** (Prop 2, MAIN).
- **Appendix completeness**: Uniqueness theorem on hierarchical coalition lattice (Prop 3, routine Shapley axiom nested).
- **Supporting** (demotable to "actionability case study"): Modality-pruned retrieval — pre-registered Stretch Goal at ≥ 10 pp downstream gain to reach venue-readiness 9+.
- **Explicit non-contributions**: 不声称 new architecture / new training objective / mechanistic interp / cross-modal alignment / long-context scaling.

## Proposed Method

### Complexity Budget

| Slot | Decision |
|---|---|
| Frozen / reused | M3-Agent retrieval / storage / generator (Qwen-2.5-VL-7B 或 InternVL) |
| New trainable | **零** |
| Excluded by design | attribution-aware retriever; SAE / probes; 3rd-level (time) Shapley; RL fine-tune; TracIn-style gradient samplers; unified MC estimator (replacing KSHAP); LoRA-gradient warm-start; default audio super-patch aggregation |

### System Overview

```
Query Q
   |
   v
M3-Agent retrieval (frozen) → M(Q) = {m_1,..,m_K}, m_k = (v_k^vision, v_k^text, v_k^audio)
   |
   +→ [Hierarchical Shapley Estimator]   ← NEW (inference-time, no training)
   |    ├ Item-level φ_k via Kernel SHAP (surrogate; ζ ≈ 50–100 utility queries)
   |    ├ Modality-level ψ_{k,ℓ} via Monte Carlo permutation (conditional on M_{¬k}),
   |    │   executed only on top-N=2 items by φ_k ranking
   |    └ Per-task aggregate conservation residual
   |
   v
Generator p_θ(y|Q, M)  (frozen)
   |
   +→ [Optional Modality-pruned Retrieval]   ← NEW (threshold operation)
   |    M̃(Q) = {v_k^(ℓ) : φ_k > τ_1  ∧  ψ_{k,ℓ} > τ_2}
   |
   v
Final answer y
```

**Hook points in M3-Agent codebase**:
- `m3_agent/retrieval.py` after retrieval return: insert attribution estimator, output sidecar tensor (φ, ψ)
- Same file, optional pruning step before generator input: switch via `--prune` flag
- No generator-side modifications. No memory-storage modifications.

### Core Mechanism

**Setup note**: Utility u(y, y*) = 1[y = y*] ∈ {0, 1}; expected utility U(S) = E_y[u(y, y*) | Q, S] ∈ [0, 1]. McDiarmid bounded-difference constant = 1.

**Algorithm**:

1. Compute item-level Shapley {φ_k}_{k=1..K} via **Kernel SHAP** with N_outer = ζ coalition samples over 2^K subset space
2. Rank items by φ_k; select top-N = 2
3. For each top-N item k, compute modality-level Shapley {ψ_{k,ℓ}}_{ℓ=1..L_k} via Monte Carlo permutation conditional on M_{¬k} frozen, N_inner = 50 samples
4. Verify global conservation residual: |Σ_k φ̂_k − [Û(M) − Û(∅)]| / |Û(M)| (per-task aggregate report)
5. **Conditional fallback**: if L_k > 6, aggregate audio token block into super-patch
6. **Optional impl optimization**: seed KSHAP coalition with Qwen-VL attention logits (with Spearman > 0.9 sanity ablation)

### Modality-Pruned Retrieval (supporting algorithm)

`M̃(Q) = {v_k^(ℓ) : φ_k > τ_1 ∧ ψ_{k,ℓ} > τ_2}`. Single threshold operation; no independent contribution.

## Theory

### Proposition 1 (Conservation)
For any query Q and any retrieved memory M = {m_1, ..., m_K} with item m_k = (v_k^(1), ..., v_k^(L_k)):

```
Σ_{k=1}^K φ_k(Q) = U(M) − U(∅)         (item-level)
Σ_{ℓ=1}^{L_k} ψ_{k,ℓ}(Q) = φ_k(Q)     (modality-level, ∀ k)
```

**Proof sketch**: Classical Shapley Efficiency axiom applied at each level of the hierarchical coalition lattice.

### Proposition 2 (Finite-sample variance reduction, MAIN)

For the hierarchical Monte Carlo estimator with N coalition samples per level:

- **2a (variance)**:
  ```
  Var(φ̂_k) ≤ σ²_max · 2^K / N
  Var(ψ̂_{k,ℓ}) ≤ σ²_max · 2^{L_k} / N
  ```
  where σ²_max ≤ 1/4 (since U ∈ [0, 1]).

- **2b (concentration via McDiarmid)**:
  ```
  P(|φ̂_k − φ_k| > ε) ≤ 2 exp(−2 N ε² / 2^K)
  ```
  (and analogously for ψ̂_{k,ℓ} with 2^{L_k}). Bounded-difference constant = 1 because u ∈ {0, 1}.

- **2c (Corollary, variance ratio bound)**:
  ```
  Var(hierarchical estimator) / Var(flat token-level estimator)  ≤  (Σ_k 2^{L_k}) / 2^{Σ_k L_k}
  ```

**Proof sketch**: bounded-differences function on each coalition + Hoeffding-McDiarmid concentration. Variance-ratio corollary follows by comparing the sub-coalition lattice sizes.

### Proposition 3 (Uniqueness, appendix only)
Among all attribution schemes over the hierarchical coalition lattice (item × modality nesting), two-level Shapley factorization is the **unique** scheme satisfying Efficiency-per-level, Symmetry-within-level, Linearity, Null-player-per-level.

**Proof sketch**: Successive application of Shapley 1953 uniqueness theorem on each level of the lattice; routine but completes the axiomatic story.

### Sample Complexity Wedge Table

| K | L_k (uniform) | Flat token Shapley | Hierarchical | Reduction |
|---|---|---|---|---|
| 2 | 2 | 2^4 = 16 | 12 | 1.3× |
| 4 | 4 | 2^16 = 65,536 | 80 | 819× |
| 4 | 6 | 2^24 = 16.7M | 272 | 61,679× |
| 8 | 4 | 2^32 = 4.3B | 384 | 11.2M× |

Wedge 在 realistic agent regime（K ∈ [4, 8], L_k ∈ [3, 6]）极显著。

### Differentiation Matrix

| Prior work | Single-shot | Hierarchical | Multimodal item | Memory-equipped | Finite-sample bound | Optimality |
|---|---|---|---|---|---|---|
| Source Attribution in RAG (2507.04480) | ✓ | ✗ | ✗ | ✗ | partial | ✗ |
| TokenShapley (2507.05261) | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Datamodels (ICLR'23) | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ |
| Conditional Shapley | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ |
| Unveiling Modality Bias (2511.05883) | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |
| MMA / M3-Agent / WorldMM | ✗ | — | ✓ | ✓ (no attr) | — | — |
| **Ours** | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Claim-Driven Validation

### Claim 1 (Main detection): M3-Bench 100-QA detection
- **Setup**: K = 4, L_k ∈ {3, 4} (vision frame + caption + audio transcript [+ scene metadata])
- **Baselines**: (i) flat token Shapley, (ii) item-only Shapley, (iii) attention-rollout, (iv) LLM self-report
- **Metrics**:
  - Separability: top vs bottom quartile prune ΔAccuracy ≥ **15 pp**
  - Conservation residual (per-task aggregate) < **5%**
  - Cost ≤ K·100 + 2·max_k L_k·50 utility queries per Q
- **Cost**: 1.6 GPU-h (variance check on 30-QA slice)

### Claim 1.1 (sanity, KSHAP seeding ablation)
- **Setup**: Compare KSHAP with Qwen-VL attention-logit seeded coalition vs uniform random coalition
- **Metric**: Spearman correlation of attribution scores > **0.9**
- **Cost**: 0.1 GPU-h (absorbed into Claim 1)

### Claim 1.5 (scaling, synthetic, validates Prop 2c)
- **Setup**: Synthetic MMA-style QA generator with **GT modality importance injected** (via masking on known-decisive modality); K ∈ {2, 4, 6, 8}, L_k ∈ {1, 2, 4, 6}
- **Metric**:
  - GT top-decile recovery ≥ **80%**
  - **Empirical variance ratio matches Corollary 2c within ±20%**
- **Cost**: 0.5 GPU-h

### Claim 1.6 (real-data LOO variance, triangulates Prop 2)
- **Setup**: M3-Bench 30 QA, vary N_samples ∈ {50, 100, 200} × 5 random seeds, compute empirical Var(φ̂)
- **Metric**: Var(φ̂) vs 1/N matches theoretical scaling within ±**30%**
- **Cost**: 0.5 GPU-h (shares forward-pass cache with Claim 1)

### Claim 2 (Supporting, demotable)
- **Setup**: M3-Bench, modality-pruned retrieval at (τ_1, τ_2) ∈ {(.05, .05), (.1, .1), (.2, .2)}
- **Baselines**: (a) **random modality-drop at matched compression** (critical control), (b) item-only pruning, (c) full retrieval
- **Promotion criterion**: modality-pruned ΔAcc ≥ 5 pp over random at 40% retention → Claim 2 stays as supporting
- **Demotion criterion**: < 5 pp → reframe as "actionability case study"
- **Stretch Goal (path to 9+)**: ≥ 10 pp ΔAcc (or ≥ 7 pp at ≤ 20% context reduction) → venue-readiness 9+
- **Pre-registered fallback**: Claims 1 + 1.5 + 1.6 carry the paper regardless

## Compute & Timeline

| Block | GPU-h |
|---|---|
| Claim 1 (incl. 1.1 seeding ablation) | 1.6 |
| Claim 1.5 synthetic scaling | 0.5 |
| Claim 1.6 real-data LOO variance | 0.5 |
| Claim 2 Pareto + random-drop baseline | 1.5 |
| Cross-benchmark generalization (WorldMM / LoCoMo) | 3.0 |
| **Total** | **7.1 GPU-h** ≤ 8 budget |

**Timeline (7 days to advisor sync)**:

| Day | Date | Task |
|---|---|---|
| 1 | 5/11 (today) | Problem statement + Prop 1 full proof |
| 2 | 5/12 | Prop 2 (variance + McDiarmid) proof draft; bounded-utility note |
| 3 | 5/13 | Prop 3 appendix proof; patch M3-Agent retrieval; implement KSHAP estimator |
| 4 | 5/14 | 30-QA toy pilot; Claim 1.1 seeding ablation; Claim 1.6 LOO data collection (shares cache) |
| 5 | 5/15 | Claim 1 (100-QA full) + Claim 1.5 synthetic |
| 6 | 5/16 | Claim 2 Pareto + variance-ratio plot + Prop 2 empirical-vs-theory table |
| 7 | 5/17 | 1-pager for advisor sync 5/18: problem statement + 2 propositions + 2 figures + Pareto curve |

## Failure Modes & Diagnostics

| Failure mode | Detection | Mitigation |
|---|---|---|
| Item-level Shapley variance 太高 | Pilot variance distribution；Var > μ red flag | Increase N (linear cost); switch KSHAP background; if still failing → **trigger backup pivot** to Cross-Modal Causal Mediation for Write-Policy |
| Conservation residual > 5% | Per-task aggregate check | Report residual as attribution-quality metric; use unbiased MC estimator (conservation in expectation) |
| Modality token count too large | L_k > 6 | Audio super-patch aggregation (preserves vision/text separability) |
| Pruned retrieval per-query drop large | Per-query Δaccuracy monitor | Fallback to full retrieval when confidence drop > δ |

## Backup direction (if pilot fails by 5/14)

**Cross-Modal Causal Mediation for Write-Policy**:
- 把 "是否在 write-time 保留 modality m" 视为 treatment, 后续 query accuracy 视为 outcome
- 用 do-calculus + interventional logits via modality masking 估计 mediated proportion
- 阈值 δ 决定写入策略
- 优点: write-policy 玩家集合离散，confounder 更可控；理论 hook 是 CMA 而非 Shapley，与 modality credit 主线略偏一档但仍 publishable

**Trigger conditions**: Claim 1 KSHAP variance > 3× theoretical bound by 5/14 toy pilot, OR conservation residual consistently > 5% across 30-QA.

## Open questions for advisor (5/18)

1. **Theory framing**: Prop 2 (finite-sample variance reduction) 作为 main theoretical claim vs Prop 3 (uniqueness) 升回 main text — 师兄偏好哪条 anchoring？
2. **Empirical bar**: 师兄之前提到"利用模型内部 signal 做 detection" — 是否倾向用 attention head / SAE feature 替代 black-box masking？若是，theoretical hook 需换。
3. **Stretch goal**: ≥ 10 pp downstream gain 是否过高？是否 5 pp + 显著 context reduction 也可接受？
4. **Target venue**: NeurIPS 2026 (Aug deadline, tight) vs ICLR 2027 (更深 evaluation 余地)？
5. **如果 Claim 2 < 5 pp**: 师兄是否接受 paper 仅以 Claim 1+1.5+1.6 为核心提交，把 Claim 2 降为 case study？

---

## Honest scoring summary

| Dimension | Final score |
|---|---|
| Problem Fidelity | 8.5 |
| Method Specificity | 9 |
| Contribution Quality | 8.5 |
| Frontier Leverage | 8 |
| Feasibility | 8 |
| Validation Focus | 8 |
| Venue Readiness | 8.5 |
| **OVERALL** | **8.5 / 10** |

**Reviewer note**: Pre-experiment ceiling ≈ 8.7. To reach 9+, **must** demonstrate ≥ 10 pp downstream gain via modality-pruned retrieval. Documented here so advisor sync has clear contingency.

---

## Skill chain — Next Steps

```
✅ /idea-creator          → 10 ideas, picked Hierarchical Modality Credit
✅ /research-refine       → THIS document (4 rounds, 8.5/10, ceiling reached)
   /novelty-check        → 下一步：deep novelty matrix on OpenReview/Semantic Scholar
   /research-review      → 拿这份给 GPT-5.4 做 standalone critical review (optional)
   /experiment-plan      → 把 7-day timeline 转成 detailed run roadmap
   implement              → 5/13 起 patch M3-Agent
   /run-experiment       → 5/15 起 M3-Bench mini-pilot
   /auto-review-loop     → 跑通后进入正式 paper 迭代
```
