# Research Idea Report — Multimodal Agent Memory × Modality Credit Attribution

**Direction**: Modality credit attribution for multimodal LLM agent memory systems
**Generated**: 2026-05-10 (deadline: 师兄 meeting 2026-05-18)
**Pipeline**: idea-creator skill → 10 ideas brainstormed (Codex o3 xhigh) → 3 eliminated by friction/novelty → 7 deep-validated → 1 final pick + 1 backup
**Pilot status**: 跳过（理论/方法层 idea，未占用 GPU；正式启动后再跑）

---

## Landscape Summary

**Memory-equipped multimodal agent**这条线在 2025–2026 极速膨胀：MMA (2602.16493) 把 memory 拆成 modality-specific storage layers + 统一 retrieval；M3-Agent (ByteDance, 2508.09736) 把 vision+audio 喂进 RL 训出来的 long-term memory 并发布 M3-Bench；WorldMM (2512.02425) 把 memory 三分为 episodic+semantic+visual 给 long video QA 用；A-Mem (NeurIPS 2025) 是 text-only Zettelkasten；MemVerse (2512.03627) 走 multimodal KG。Survey 已成型："Memory in LLMs" (2509.18868)、"Memory for Autonomous LLM Agents" (2603.07670)，ICLR 2026 还有 MemAgents workshop。

**Modality importance / attribution**这条线主要落在 **single-shot multimodal classification**：Unveiling Modality Bias (2511.05883) 用 Shapley + causal mediation 给 misinformation 分类做 modality credit；Disentangling Bias via Causal Attention (2508.04999) 在 sentiment 上做 intra/inter-modal CMA；MMEL 用 hierarchical gradient attribution；ICLR 2025 Blogposts 系统综述了 mechanistic interp for VLM（probing / activation patching / logit lens / SAE）；Circuit tracing in VLMs (2602.20330) 给 image residuals 的 attribution graph。

**Per-document Shapley for RAG**这块**已经饱和**：Source Attribution in RAG (2507.04480) 走 Kernel SHAP；ContextCite 走 surrogate；TokenShapley (2507.05261) 走 token 粒度；MaxShapley (2512.05958)、Fair Document Valuation in Summaries (2505.23842)、RepoShapley (2601.03378) 都是变体。**这是 idea 选型的关键约束**：任何"对 retrieved memory 做 Shapley"如果不显式利用 (item × modality × time) 的层次结构和 agent loop 的动态 retrieval pool，会被 reviewer 直接 discount。

**关键白区**：modality credit attribution 在 **memory-equipped multi-step agent** 上是空的——决策条件于 retrieved memory items，每个 memory item **自身就是多模 composition**（episode = video frame + caption + audio transcript），且 retrieval pool 随 query 动态变化。Existing Shapley/CMA paper 都假定 single-shot、玩家集合静态、玩家无内部模态结构。

---

## Recommended Ideas (ranked)

### ★ FINAL PICK — Idea 1: Hierarchical Modality Credit for Multimodal Agent Memory

- **Memory component targeted**: Retrieval（一级）+ Cross-modal binding（二级 item-内部归因）
- **Core scientific question**: 在多步 multimodal memory agent 中，如何把一次 answer 的 credit **同时**分摊到 (memory item × modality × time)，且保持 hierarchical consistency？
- **Method sketch**:
  1. 把一次 inference 视作 hierarchical cooperative game：top level 玩家 = 检索到的 memory items $\{m_1,\dots,m_K\}$；bottom level 玩家 = item 内部的 modality patches $\{v_k^{(1)},\dots,v_k^{(L_k)}\}$（vision/text/audio token blocks）。
  2. 用 one-step masking + Kernel SHAP 估计 item-level Shapley $\phi_k$，再在 high-$\phi$ item 内部估计 modality-level Shapley $\psi_{k\ell}$。
  3. 为 dynamic retrieval pool 引入 **conditional Shapley**：玩家集合是 query-conditioned $\mathcal{M}(Q)$，需在估计时显式 marginalize over retrieval randomness。
  4. Detection→Improvement 自然延伸：基于 $\phi_k, \psi_{k\ell}$ 做 **modality-pruned retrieval**——保留高贡献 modality patch，evict 低贡献的，验证 token cost 与 accuracy 的 trade-off。
- **Theoretical hook** (formal):
  - 定义 Hierarchical Shapley value 并证明 **conservation**:
    $\sum_k \phi_k = u(\mathcal{M}) - u(\emptyset)$ 且 $\sum_\ell \psi_{k\ell} = \phi_k$
  - 给出 finite-sample estimator 的 $\mathcal{O}(1/\sqrt{N})$ 误差界
  - **关键差异化命题**：在 dynamic retrieval pool 下，naive document-level Shapley **不再** efficient（玩家数 × modality 爆炸）；hierarchical decomposition 把样本复杂度从 $\mathcal{O}(2^{KL})$ 降到 $\mathcal{O}(2^K + K \cdot 2^L)$
- **Empirical pilot** (≤ 2 GPU-h): M3-Bench 100-QA subset，K=4 retrieved memory items，每 item 含 video frame + caption + audio。Metric: 把 attribution 高/低分位 prune 后的 accuracy drop 差 > 15 pp 即成功。
- **Novelty risk** (sharpened):
  - **Closest prior**: Source Attribution in RAG (2507.04480) — per-document Shapley，**but** 单模、static pool、无 item-internal hierarchy
  - **Differentiation**: 我们的玩家集合 = (item, modality, time) 三元组而非单纯 document；retrieval pool 是 query-dependent 而非 fixed context；attribution target 是 **agent decision over time** 而非 single answer
  - **如果 reviewer 问"为什么不能直接套用 Shapley/CMA paper？"**: 因为 memory agent 的玩家集合随 query 动态生成，且 memory item 内含多模 patch；直接套用既无法 hierarchically 分摊 credit，也无法 preserve efficiency
- **Detection→Improvement extension**: ✅ Modality-pruned retrieval 直接由 attribution 排序导出
- **Friction score**: 2 (M3-Agent 开源，masking 实现成熟，理论可写完整)
- **Risk**: LOW

**Formal Problem Statement (LaTeX-ready, take-to-advisor draft)**:

```latex
\textbf{Setup.} Let $Q$ denote a query, $\mathcal{M}(Q) = \{m_1,\dots,m_K\}$
the query-conditioned retrieved memory, where each item
$m_k = (v_k^{(1)},\dots,v_k^{(L_k)})$ is a tuple of modality patches
(vision, text, audio). Let $p_\theta(y \mid Q, \mathcal{M})$ be the agent's
output distribution and $u(y, y^\ast) = \mathbf{1}[y=y^\ast]$ the utility.

\textbf{Hierarchical Shapley value.}
\emph{Item level:}
\[
\phi_k(Q) = \sum_{S \subseteq \mathcal{M}\setminus\{m_k\}}
  \frac{|S|!\,(K-|S|-1)!}{K!}
  \bigl[U(S \cup \{m_k\}) - U(S)\bigr],
\]
where $U(S) = \mathbb{E}_y[u(y,y^\ast) \mid Q, S]$.

\emph{Modality level (within item $k$):}
\[
\psi_{k\ell}(Q) = \sum_{T \subseteq m_k\setminus\{v_k^{(\ell)}\}}
  \frac{|T|!\,(L_k-|T|-1)!}{L_k!}
  \bigl[U_{\setminus k}(T \cup \{v_k^{(\ell)}\}) - U_{\setminus k}(T)\bigr],
\]
where $U_{\setminus k}(T) = \mathbb{E}_y[u(y,y^\ast) \mid Q,
\mathcal{M}_{\setminus k} \cup T]$.

\textbf{Conservation (Proposition 1, to prove).}
\[
\sum_{k=1}^K \phi_k = U(\mathcal{M}) - U(\emptyset),
\qquad
\sum_{\ell=1}^{L_k} \psi_{k\ell} = \phi_k.
\]

\textbf{Sample complexity (Proposition 2).} Naive joint Shapley over
$(item, modality)$ requires $\mathcal{O}(2^{\sum_k L_k})$ utility queries.
Hierarchical decomposition reduces to
$\mathcal{O}(2^K + \sum_k 2^{L_k})$, with estimator error
$|\hat\phi_k - \phi_k| = \mathcal{O}(1/\sqrt{N})$ for $N$ Monte-Carlo samples.

\textbf{Improvement algorithm.} Given thresholds $\tau_1, \tau_2$, define
modality-pruned retrieval $\tilde{\mathcal{M}}(Q) = \{v_k^{(\ell)} :
\phi_k > \tau_1 \land \psi_{k\ell} > \tau_2\}$ and conjecture
$U(\tilde{\mathcal{M}}) \approx U(\mathcal{M})$ at $|\tilde{\mathcal{M}}|
\ll |\mathcal{M}|$.
```

---

### ★ BACKUP — Idea 2: Cross-Modal Causal Mediation for Write-Policy

- **Memory component**: Write-policy (which modality of an experience to actually store)
- **Core question**: 写入 episode 时，哪种 modality 是真正的 **mediator**（而非 confounder）决定后续任务准确率？
- **Method sketch**: 把"是否在写入时保留 modality m"视为 treatment，把后续 query accuracy 视为 outcome。用 causal mediation analysis (do-calculus + interventional logits via modality masking) 估计 total / direct / mediated effect。基于 mediated proportion 决定写入策略。
- **Theoretical hook**: 若 mediated proportion > $\delta$，则 omit modality $m$ 至少损失 $\delta \cdot \Delta\text{Acc}$。给出 mediated-effect 下的 write-policy 最优性条件。
- **Empirical pilot**: ScienceQA-with-history small split，$\delta = 0.2$ threshold，p < 0.05 显著。
- **Novelty risk**: CMA 在 single-shot 多模 paper 已出现 (2511.05883, 2508.04999)；novelty 在于 **write-time decision** 而非 inference-time bias。中等 novelty。
- **Detection→Improvement**: ✅ Skip writes with mediated effect < ε to save memory budget
- **Friction**: 3, **Risk**: MEDIUM
- **Why backup**: 理论深度同样硬（CMA），但需要可控 confounder 假设；如果 ① 在 pilot 阶段噪声太大无法 separate item-vs-modality contributions，可立即 fallback 到 ④（write-policy 比 retrieval 更可控）。

---

### Other survivors (not picked but on shelf)

| # | Title | Component | Friction | Why deprioritized |
|---|-------|-----------|----------|-------------------|
| 3 | Causal Forgetting Influence Functions | Forgetting | 3 | Hessian-vector 在 MLLM 上不可靠；prior 已有 "Remember to Forget" (ICLR '25) |
| 4 | IB Cross-Modal Binding | Binding | 3 | IB 老掉牙；MI estimator 不稳；理论故事不够独立 |
| 5 | Temporal-Decay Shapley Consolidation | Consolidation | 2 | SimpleMem / Agentic Plan Caching 已做 consolidation；Shapley + decay 增量小 |
| 6 | Sequential Shapley Dropout Encoder | Encoding | 3 | "TokenGrad Dropout" (ACL '25) prior 风险 |
| 7 | Double-Descent Retrieval Depth | Retrieval depth | 2 | Findings EMNLP '25 "Retrieval-Depth Trade-off in RAG" 已做文本版；多模扩展太薄 |

---

## Eliminated Ideas (initial filter)

| Idea | Reason eliminated |
|------|-------------------|
| Item-And-Time Influence Graphs | Friction 4，需要 graph infra，1 周内做不出 toy |
| Memory-Aware Modality Masking Noise | 与 IMD / Dynamic Modality Scheduling 撞线 |
| Surprise-Triggered Eviction | RL memory 已有 surprise-based eviction，novelty 弱 |

---

## Reviewer Objections & Mitigations (for Final Pick ①)

| Reviewer 视角 | Objection | Mitigation |
|---|---|---|
| Methodological | "只是 Shapley 套两层 hierarchy，方法增量小" | 强调 **conditional Shapley over dynamic retrieval pool** + sample-complexity reduction theorem，而非简单分层 |
| Empirical | "K 太小时 marginal Δlog-prob 噪声大，分位曲线分不开" | Pilot 用 K=8 + Kernel SHAP surrogate；预先报告 estimator variance bound |
| Novelty | "Source Attribution in RAG (2507.04480) 已经做了 per-document Shapley" | 我们 differentiate: 单模 vs 多模 hierarchy；static context vs dynamic agent loop；single-shot vs multi-step decision |
| Theory | "Conservation 命题在 standard Shapley 是 trivial" | 我们的核心是 **hierarchical** + **dynamic pool** 下的 conservation 与 sample-complexity bound；后者非平凡 |

---

## Suggested Execution Order (1 week to advisor meeting)

**Day 1–2 (now → 5/12)**:
- 把上面的 Formal Problem Statement 抄进 `~/ideas/jeac_modality_credit_problem.md`（或新开 file），打磨数学符号
- 跑通 ContextCite 或 Kernel SHAP 在 single-modal RAG 上的开源实现，确认 utility-call 的开销
- 读 M3-Agent repo，定位 retrieval API 的 hookable 点

**Day 3–4 (5/13 → 5/14)**:
- 实现 item-level Shapley masking（先不做 modality 内部）
- 在 M3-Bench 抽 30-QA mini-pilot，看 attribution score 是否区分 high/low 分位

**Day 5–6 (5/15 → 5/16)**:
- 加入 modality-level Shapley
- 写出 conservation 命题的 formal proof（standard Shapley 性质 + hierarchical inductive 即可）
- Sample-complexity bound 的草稿

**Day 7 (5/17)**:
- 整理：1 页 problem statement + 1 张 attribution heatmap + 1 个 toy table（pruned vs full retrieval）
- 5/18 跟师兄聊：核心讨论点是"哪个 component 最值得攻"——拿这份给他选

---

## Open questions for advisor (5/18)

1. 师兄之前提的"利用模型内部 signal 做 detection"——是想用 attention head / SAE feature / logit lens 这类 mechanistic interp signal 替代 masking-based attribution 吗？如果是，theoretical hook 要换。
2. 是否接受 ① 的 framing（retrieval × cross-modal）vs 改走 ②（write-policy mediation）？前者实验更现成，后者理论更"causal"。
3. Target venue：NeurIPS 2026 (Aug deadline) vs ICLR 2027？前者紧，后者可做更深。
4. Compute budget 上限是多少？M3-Bench full eval 需要 ~10 GPU-h；mini pilot 2 GPU-h 内可解。

---

## Skill chain — Next Steps

```
✅ /idea-creator          ← done (this report)
   /research-refine       ← 下一步：把 ① 的 formal statement 打磨成完整 method plan
   /novelty-check         ← 对 final pick 跑深度 novelty (跨 OpenReview/Semantic Scholar)
   /research-review       ← 拿 problem statement 给 GPT-5.4 xhigh 做 critical review
   implement              ← 5/13 起 patch M3-Agent
   /run-experiment        ← M3-Bench mini-pilot
   /auto-review-loop      ← 跑通后进入正式迭代
```

---

## Sources

**Memory-equipped multimodal agent**
- [MMA: Multimodal Memory Agent](https://arxiv.org/abs/2602.16493) — 师兄分享的论文
- [M3-Agent / Seeing, Listening, Remembering, Reasoning](https://arxiv.org/abs/2508.09736)
- [WorldMM: Dynamic Multimodal Memory Agent](https://arxiv.org/html/2512.02425v1)
- [A-Mem: Agentic Memory for LLM Agents (NeurIPS 2025)](https://openreview.net/forum?id=FiM0M8gcct)
- [MemVerse: Multimodal Memory for Lifelong Learning](https://arxiv.org/html/2512.03627v1)
- [M2A: Dual-Layer Hybrid Memory](https://arxiv.org/html/2602.07624v1)
- [Memory in LLMs Survey](https://arxiv.org/pdf/2509.18868)
- [Memory for Autonomous LLM Agents Survey](https://arxiv.org/html/2603.07670v1)
- [ICLR 2026 MemAgents Workshop Proposal](https://openreview.net/pdf?id=U51WxL382H)

**Modality importance / attribution (single-shot)**
- [Unveiling Modality Bias (Shapley + CMA)](https://arxiv.org/html/2511.05883)
- [Disentangling Bias via Intra/Inter-modal Causal Attention](https://arxiv.org/html/2508.04999)
- [Mechanistic Interp Meets VLMs](https://d2jud02ci9yv69.cloudfront.net/2025-04-28-vlm-understanding-29/blog/vlm-understanding/)
- [Circuit Tracing in VLMs](https://arxiv.org/html/2602.20330v1)
- [Multi-Modal Interpretability for Localization in VLMs](https://arxiv.org/abs/2509.15243)

**Per-document Shapley for RAG (saturated; novelty boundary)**
- [Source Attribution in RAG (Kernel SHAP)](https://arxiv.org/abs/2507.04480)
- [TokenShapley](https://arxiv.org/html/2507.05261v2)
- [MaxShapley](https://arxiv.org/html/2512.05958)
- [Fair Document Valuation in LLM Summaries](https://arxiv.org/html/2505.23842)
- [Document Attribution in RAG (Inria 2025)](https://project.inria.fr/aimlai/files/2025/09/02-AIMLAI2025.pdf)

**Modality dropout / robustness**
- [Dynamic Modality Scheduling](https://www.arxiv.org/pdf/2506.12724)
- [MMP: Masked Modality Projection](https://arxiv.org/html/2410.03010v1)
