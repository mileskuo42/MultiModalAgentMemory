# Owen-Value Attribution for Multimodal Agent Memory
## 实验框架搭建文档 (Experimental Framework Setup)

**Project codename**: `modality_credit`
**Owner**: Moyang Guo (Northwestern PhD, advised by 师兄)
**Advisor sync**: 2026-05-18
**Start of implementation**: 2026-05-13
**Target venues**: NeurIPS 2026 D&B / CVPR 2026 application / ICLR 2027
**Total compute budget**: ~26 GPU-h (含 30% debug buffer over 20 GPU-h baseline)

---

## 目录

1. [项目介绍](#1-项目介绍)
2. [Motivation](#2-motivation)
3. [Method 简要回顾](#3-method-简要回顾)
4. [Repository 结构](#4-repository-结构)
5. [环境配置 & 依赖](#5-环境配置--依赖)
6. [核心模块设计](#6-核心模块设计)
7. [Pre-pilot Audits（5/13 必做）](#7-pre-pilot-audits513-必做)
8. [Phase-by-phase 实验脚本](#8-phase-by-phase-实验脚本)
9. [数据 Pipeline](#9-数据-pipeline)
10. [Baselines 实现](#10-baselines-实现)
11. [Adversarial Generator (Phase 4A)](#11-adversarial-generator-phase-4a)
12. [Logging & WandB](#12-logging--wandb)
13. [Compute Deployment](#13-compute-deployment)
14. [Risk Monitoring & Decision Tree](#14-risk-monitoring--decision-tree)
15. [实施时间线](#15-实施时间线)
16. [相关文档](#16-相关文档)

---

## 1. 项目介绍

### 1.1 一句话

第一次把 **Owen value (cooperative game theory, 1977)** 落到 multimodal memory-equipped agent 的归因问题上：对每个 query Q 和 retrieved memory M = {m₁, ..., m_K}（每条 item 含 L_k 个 modality patch），算出 **per-(item × modality) 的 query-conditioned support weight**，并用这些 weight 做：
1. **失败 diagnosis**（哪条 memory 的哪种 modality 误导了 generator）
2. **modality-pruned retrieval**（attribution-driven context 压缩，目标 ΔAcc ≥ 10pp at ≤40% retention）

### 1.2 三层 contribution

| Contribution | 内容 | 在 paper 里的位置 |
|---|---|---|
| **C1 · Application** | First Owen-value × multimodal memory agent | Abstract + Intro |
| **C2 · Algorithm** | Closed-form sample allocation N_out* ∝ √(2^K), N_in^(k)* ∝ √(2^{L_k}) | Method (Algorithm 1) |
| **C3 · Empirical** | Modality-pruned retrieval ≥ 10pp ΔAcc | Experiment (Claim 2, must-accept hook) |

### 1.3 为什么这个 setting 对 Owen value 特别合适

Owen / Shapley 要求 **permutation invariance**——u(S) 对任意子集 S 都要 well-defined。

- RL credit (OSPO) 受困于 trajectory 的因果时序，无法 freely mask 中间 action
- 我们的 retrieval 把 agent 的动态因果流切成**静态 snapshot 集合**——K 条 item 在 generator context 里并行 attend，可以自由 mask 任意 (item × modality) 子集
- memory 的真正时序在 write-time，那一层有因果时序问题——我们主动切割那层 (non-goal)，留给 backup direction

### 1.4 相对 MMA / ContextCite / MM-SHAP 的定位

| 范式 | 代表 | 输入 | 输出 |
|---|---|---|---|
| Instance-independent trust prior | MMA (PKU, 2026) | M_i | per-item trust |
| Query-conditioned support, atomic | ContextCite, Source-Attr-in-RAG | (Q, M) | per-item support |
| **Query-conditioned support, hierarchical** ← 我们 | Owen attribution | (Q, M) | **per-(item × modality) support** |

我们的 utility:

```
U(S) = E_y[ 1[y = y*] | Q, S ]  =  P(model 答对 | query Q, context = subset S)
```

**Query-conditioned 但 not 条件在具体生成的 y 上**——和 ContextCite 同一范式（perturb context, measure marginal effect on correctness），但 player 粒度推到 (item × modality)。

---

## 2. Motivation

### 2.1 场景

2025–2026 涌现了 **memory-equipped multimodal agent**（M3-Agent / WorldMM / MMA / MemVerse），让 agent 在长期交互中持续看视频、听音频、读对话，把这些观察以 episode 形式存进 long-term memory。用户提问时从 memory 里 retrieve K 条相关 episode，喂给 base VLM (Qwen-2.5-VL) 生成答案。

每条 episode 是多模 composition：

```
m_k = ( v_k^vision,  v_k^text,  v_k^audio  [+ scene metadata] )
```

K 通常 4–8，L_k 通常 3–5。

### 2.2 痛点

当 agent 答错时，开发者要知道：

- 是 retrieval 错了——拉出了不相关的 memory？
- 还是 retrieval 对了但 memory 里某个 modality 误导了 generator？
- 还是某条 memory 的 audio 转录 hallucination 把 generator 带跑偏？

三个失败模式需要完全不同的修复策略，但当前 agent 的归因能力是**全瞎的**。最常见做法是 black-box masking——挨条去掉 memory 看 accuracy 变化。但这只能告诉你"去掉第 3 条 memory 准确率掉 4pp"，**回答不了"是第 3 条 memory 里的哪个 modality 起了关键作用"**。

### 2.3 现有工作差在哪：5 条路径

| Path | 代表 | 缺什么 |
|---|---|---|
| 1. Per-document Shapley | Source-Attr-in-RAG, ContextCite | atomic player, 无 modality 内部结构 |
| 2. Token-level Shapley | TokenShapley | 玩家爆炸 (≥200 token at K=4), variance 失控 |
| 3. Attention / IG | MMEL, CAGUL | 无 conservation, 难可靠解释 |
| 4. Memory arch | M3-Agent, WorldMM, MMA | 根本不做归因 |
| 5. Single-shot modality | MM-SHAP, MultiSHAP, Park MIS | 无 memory hierarchy, 单次预测 |

我们填的 4 维交集（Owen × multimodal × memory agent × inference-time）经 deep novelty check (2026-05-11) 已确认无人做过。

### 2.4 为什么现在做

memory-equipped multimodal agent 在 2024 之前是工程稀有品种；2025 之后才有可用开源 codebase（M3-Agent / WorldMM 都是 2025-08 / 2025-12）。归因社区工具链此前主要服务 single-shot classification 或 text-only RAG，没人落到这个 niche。窗口期正好。

---

## 3. Method 简要回顾

详细版见 [`multimodal_memory_credit_method_plan_repositioned.md`](multimodal_memory_credit_method_plan_repositioned.md)。这里只列实现需要用到的核心定义。

### 3.1 两个 game

```
Outer game G_out:   players = {m_1, ..., m_K} (K items)
                    utility = U(S) = P(model 答对 | Q, S ⊆ M)
                    output  = {φ_k}_{k=1..K}

Inner game G_in^(k): players = modalities of m_k
                    utility = U(S_k ∪ M_{¬k} 全保留 | Q)
                              即把其他 item 全部冻结, 只 mask m_k 内部的 modality
                    output  = {ψ_{k,ℓ}}_{ℓ=1..L_k}
```

### 3.2 Owen value 计算流程

1. 用 **Kernel SHAP** 估 outer φ_k（采样 N_out 个 coalition）
2. 按 φ_k 排序选 top-N=2 item
3. 对每个 top-N item，用 **MC permutation Shapley** 估 inner ψ_{k,ℓ}（采样 N_in^(k) 个 permutation, conditional on M_{¬k} frozen）
4. （可选）seeding KSHAP coalition with Qwen-VL attention logits

### 3.3 Algorithm 1 — 闭式采样配置

```
Given:    K, {L_k}_{k=1..K}, total utility-query budget B, top_n_inner
Compute:  Z = √(2^K) + Σ_{k ∈ TopN} √(2^{L_k})
          N_out*       = floor( B · √(2^K) / Z )
          N_in^(k)*    = floor( B · √(2^{L_k}) / Z )   for each k ∈ TopN
```

where TopN ⊆ {1..K} has size `top_n_inner` (default 2) — only those items
get inner-gamed, so the denominator must include only those terms.
Initial draft used `Σ_{k=1..K} √(2^{L_k})` and under-spent budget; see issue #2.

### 3.4 Theory（Prop 2）

```
Var(φ̂_k)         ≤ 2^K / (4N)
Var(ψ̂_{k,ℓ})     ≤ 2^{L_k} / (4N)
P(|φ̂_k − φ_k| > ε)   ≤ 2 exp(−2Nε² / 2^K)      [McDiarmid]
Var(hier) / Var(flat) ≤ (Σ 2^{L_k}) / 2^{Σ L_k}  [Cor 2c]
```

### 3.5 Modality-pruned retrieval

```
M̃(Q) = { v_k^(ℓ) : φ_k > τ_1  ∧  ψ_{k,ℓ} > τ_2 }
```

---

## 4. Repository 结构

```
modality_credit/
├── README.md
├── pyproject.toml             # poetry / uv
├── requirements.txt
├── .env.example                # API keys (Qwen-VL, OpenAI for baselines, WandB)
├── Makefile                    # 常用 command alias
│
├── configs/                    # 所有实验 config (Hydra-style YAML)
│   ├── base.yaml
│   ├── phase0_audit.yaml
│   ├── phase1_toy_pilot.yaml
│   ├── phase2_detection.yaml
│   ├── phase3_pruning.yaml
│   ├── phase4a_poisoning.yaml
│   ├── phase4b_longvideobench.yaml
│   └── phase4e_human_study.yaml
│
├── data/                       # 原始 + 处理后数据
│   ├── m3_bench/
│   │   ├── raw/                # 直接 clone M3-Agent repo 下载
│   │   ├── processed/          # 我们 preprocess 后的 (Q, M, y*) tuples
│   │   └── splits.json         # train/val/test split
│   ├── longvideobench/
│   ├── worldmm/
│   ├── mma/
│   ├── synthetic/              # Claim 1.5 合成数据
│   │   ├── generator.py
│   │   └── samples/
│   └── adversarial/            # Phase 4A 注毒数据
│       └── samples/
│
├── modality_credit/            # 主代码 package
│   ├── __init__.py
│   ├── utility.py              # U(S) — core abstraction
│   ├── masking.py              # mask vision/text/audio 实现
│   ├── m3_agent_adapter.py     # 桥接 M3-Agent codebase
│   │
│   ├── estimator/
│   │   ├── __init__.py
│   │   ├── kernel_shap.py      # outer game (KSHAP)
│   │   ├── mc_permutation.py   # inner game (MC perm)
│   │   ├── owen.py             # 调度 outer + inner
│   │   └── allocation.py       # Algorithm 1
│   │
│   ├── pruning.py              # Algorithm 3 (M̃ computation)
│   │
│   ├── baselines/
│   │   ├── flat_shapley.py
│   │   ├── item_only.py
│   │   ├── attention_rollout.py
│   │   ├── llm_self_report.py
│   │   └── mma_reliability.py  # MMA C(M_i) baseline
│   │
│   ├── adversarial/
│   │   ├── caption_flip.py
│   │   ├── audio_noise.py
│   │   ├── frame_swap.py
│   │   └── entity_swap.py
│   │
│   ├── audits/                 # Pre-pilot sanity checks
│   │   ├── modality_leakage.py
│   │   ├── u_empty_baseline.py
│   │   ├── positional_invariance.py
│   │   └── conservation_residual.py
│   │
│   ├── metrics/
│   │   ├── separability.py     # top-vs-bottom ΔAcc
│   │   ├── auroc.py            # poisoning detection
│   │   └── pareto.py           # retention-accuracy curve
│   │
│   └── utils/
│       ├── seeding.py          # KSHAP attention-logit seeding
│       ├── logging.py          # WandB wrapper
│       └── io.py               # save/load tensor sidecars
│
├── scripts/                    # 每个 phase 的可执行脚本
│   ├── 00_setup_m3_agent.sh
│   ├── 01_pre_pilot_audit.py
│   ├── 02_toy_pilot.py
│   ├── 03_full_detection.py
│   ├── 04_pruning_pareto.py
│   ├── 05_cross_architecture.py
│   ├── 06_poisoning_detection.py
│   ├── 07_longvideobench_k8.py
│   └── 08_human_study_prep.py
│
├── results/                    # 实验产出
│   ├── phase0_audit/
│   ├── phase1_pilot/
│   ├── phase2_detection/
│   ├── phase3_pruning/
│   └── phase4_*/
│
├── notebooks/                  # 分析 + 可视化
│   ├── 01_pre_pilot_audit.ipynb
│   ├── 02_theory_validation.ipynb
│   ├── 03_pareto_curves.ipynb
│   ├── 04_failure_taxonomy.ipynb
│   └── 05_human_study_analysis.ipynb
│
└── tests/                      # unit tests
    ├── test_owen_estimator.py
    ├── test_allocation.py
    ├── test_masking.py
    └── test_conservation.py
```

---

## 5. 环境配置 & 依赖

### 5.1 系统要求

- Python 3.10+
- CUDA 12.1+
- 1× A100 80GB (或 4× A6000 48GB)
- 60GB disk for M3-Bench full + LongVideoBench

### 5.2 主要依赖（`pyproject.toml`）

```toml
[project]
name = "modality_credit"
version = "0.1.0"
dependencies = [
    "torch>=2.3.0",
    "transformers>=4.45.0",
    "qwen-vl-utils>=0.0.8",          # Qwen-2.5-VL 工具
    "shap>=0.46.0",                   # Kernel SHAP 参考实现
    "numpy>=1.26",
    "scipy>=1.13",
    "pandas>=2.2",
    "scikit-learn>=1.5",              # AUROC, clustering
    "hydra-core>=1.3",                # config management
    "wandb>=0.17",
    "pillow>=10.0",                   # frame loading
    "decord>=0.6",                    # video loading
    "librosa>=0.10",                  # audio
    "openai-whisper>=20231117",       # audio transcription if needed
    "tqdm",
    "rich",                            # 美化 logging
    "pytest>=8.0",
]

[project.optional-dependencies]
human_study = ["streamlit>=1.36", "annotation-server"]
deploy = ["modal>=0.64", "vastai>=0.2"]
```

### 5.3 关键 API keys（`.env`）

```bash
WANDB_API_KEY=...
HF_TOKEN=...                # huggingface for Qwen-VL / InternVL / LLaVA weights
OPENAI_API_KEY=...          # LLM self-report baseline
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
```

### 5.4 安装步骤

```bash
git clone <our-repo> modality_credit && cd modality_credit
git submodule add https://github.com/bytedance-research/M3-Agent vendor/m3_agent
git submodule add https://github.com/AIGeeksGroup/MMA vendor/mma
git submodule add https://github.com/<worldmm-repo> vendor/worldmm

# 用 uv 装（比 pip 快 10×）
pip install uv
uv venv && source .venv/bin/activate
uv pip install -e .

# 下载 base model
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/qwen-vl

# Verify install
python -c "from modality_credit.utility import build_utility; print('OK')"
pytest tests/ -v
```

---

## 6. 核心模块设计

### 6.1 `utility.py` — Owen estimator 的核心抽象

整个 framework 的中心。其他模块都通过 `U(S)` 调用 generator，所有 attribution 都靠这一个函数。

```python
# modality_credit/utility.py
from dataclasses import dataclass
from typing import Literal
import torch

Modality = Literal["vision", "text", "audio", "scene"]

@dataclass
class MemoryItem:
    """Represents one retrieved episode."""
    item_id: str
    modalities: dict[Modality, object]   # e.g., {"vision": frame_tensor, "text": "切番茄", "audio": "刀切声"}
    metadata: dict = None                # 时间戳, source confidence 等

@dataclass
class QueryInstance:
    query: str
    memory: list[MemoryItem]
    gold_answer: str                     # used for u(y, y*) = 1[y = y*]
    
class Utility:
    """
    Core abstraction: U(S) = P(model 答对 | Q, S)
    
    Args:
        generator: callable (query, context_str) -> generated_answer
        verifier: callable (generated, gold) -> bool (exact-match or LLM-judge)
    """
    def __init__(self, generator, verifier, cache_dir=None):
        self.generator = generator
        self.verifier = verifier
        self.cache = {}  # 关键: 同一 subset 多次查询不重算
        self.cache_dir = cache_dir
    
    def evaluate(self, query_inst: QueryInstance, item_mask: list[bool], 
                 modality_masks: list[dict[Modality, bool]] = None) -> float:
        """
        Compute U(S).
        
        item_mask:        list of K bools, True = include item
        modality_masks:   list of K dicts, mask within each item (None = include all)
        
        Returns: u(y_generated, y_gold) ∈ {0, 1}
        """
        cache_key = self._make_key(query_inst, item_mask, modality_masks)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        context = self._build_context(query_inst.memory, item_mask, modality_masks)
        y = self.generator(query_inst.query, context)
        u = float(self.verifier(y, query_inst.gold_answer))
        
        self.cache[cache_key] = u
        return u

    def _build_context(self, memory, item_mask, modality_masks):
        """Build context string/tensor by applying masks."""
        from modality_credit.masking import apply_masks
        return apply_masks(memory, item_mask, modality_masks)

    def _make_key(self, q, item_mask, mod_mask):
        # 哈希以避免重复 forward
        ...

def build_utility(model_name="qwen-vl-7b", verifier="exact_match") -> Utility:
    """Factory for default Qwen-VL utility."""
    from modality_credit.m3_agent_adapter import load_generator
    gen = load_generator(model_name)
    ver = _load_verifier(verifier)
    return Utility(gen, ver)
```

**关键设计**:
- `Utility.evaluate()` 是 framework 唯一的"昂贵调用"——每次 = 1 次 generator forward pass
- **必须有 cache**：Shapley 算法在同一个 (Q, M) 上反复算不同 S，同一 S 不应重算
- Verifier 默认 exact match；M3-Bench 是开放问答，可能需要 LLM-judge fallback

### 6.2 `masking.py` — Modality mask 实现

实现"删 modality"这一动作。必须真的删干净，否则 attribution 失真（见 [Pre-pilot Audit 7.1](#71-modality-leakage-audit)）。

```python
# modality_credit/masking.py
from PIL import Image
import torch

def apply_masks(memory, item_mask, modality_masks=None):
    """
    Build the generator context from memory under item/modality masking.
    """
    out_blocks = []
    for k, item in enumerate(memory):
        if not item_mask[k]:
            continue                            # item 整个去掉
        kept_modalities = item.modalities.copy()
        if modality_masks is not None:
            for m, keep in modality_masks[k].items():
                if not keep:
                    kept_modalities[m] = _redact(m, item.modalities[m])
        out_blocks.append(format_item(item.item_id, kept_modalities))
    return "\n\n".join(out_blocks)

def _redact(modality, content):
    """Replace content with a neutral placeholder so generator sees the modality is absent."""
    if modality == "vision":
        # Black-out the frame (constant tensor) instead of removing the position
        # Q: should we just drop the image token? 见 audit
        return _black_frame_like(content)
    elif modality == "text":
        return "[caption redacted]"
    elif modality == "audio":
        return "[audio redacted]"
    elif modality == "scene":
        return "[scene metadata redacted]"

def _black_frame_like(frame):
    if isinstance(frame, Image.Image):
        return Image.new("RGB", frame.size, (0, 0, 0))
    elif isinstance(frame, torch.Tensor):
        return torch.zeros_like(frame)

def format_item(item_id, modalities):
    """Render a single retrieved item into a string block fed to Qwen-VL."""
    parts = [f"=== Memory item {item_id} ==="]
    for m, c in modalities.items():
        if m == "vision":
            parts.append(f"[image] (see attached frame)")
        elif m == "text":
            parts.append(f"[caption] {c}")
        elif m == "audio":
            parts.append(f"[audio transcript] {c}")
        elif m == "scene":
            parts.append(f"[scene] {c}")
    return "\n".join(parts)
```

**关键 design decision**:
- **Redaction vs Removal**：我们选 redaction（替换成 placeholder），保留 position 信息。如果 removal 会让 K-1 个 item 在 attention 模式上和 K 个完全不同 → 引入 spurious effect
- Vision 用 black frame 替换而不是删除——保留 visual token 数量
- Audit 7.1 必须验证 redaction 真的让 generator 看不到原信息

### 6.3 `estimator/owen.py` — Owen value 主调度

```python
# modality_credit/estimator/owen.py
import numpy as np
from .kernel_shap import kernel_shap_outer
from .mc_permutation import mc_perm_inner
from .allocation import allocate_budget

class OwenEstimator:
    def __init__(self, utility, top_n_inner=2, seed=42):
        self.utility = utility
        self.top_n = top_n_inner
        self.seed = seed
    
    def estimate(self, query_inst, budget_B=500):
        K = len(query_inst.memory)
        L_ks = [len(item.modalities) for item in query_inst.memory]
        
        # Algorithm 1: allocate budget
        N_out, N_in = allocate_budget(K, L_ks, budget_B)
        
        # Outer game
        phi = kernel_shap_outer(
            utility=self.utility,
            query_inst=query_inst,
            N=N_out,
            seed=self.seed,
        )
        
        # Pick top-N items for inner game
        top_idx = np.argsort(-phi)[: self.top_n]
        
        # Inner game per top item
        psi = np.full((K, max(L_ks)), np.nan)
        for k in top_idx:
            psi_k = mc_perm_inner(
                utility=self.utility,
                query_inst=query_inst,
                item_idx=k,
                N=N_in[k],
                seed=self.seed + k,
            )
            psi[k, :len(psi_k)] = psi_k
        
        # Conservation check
        U_full = self.utility.evaluate(query_inst, [True]*K)
        U_empty = self.utility.evaluate(query_inst, [False]*K)
        conservation_residual = abs(phi.sum() - (U_full - U_empty)) / max(abs(U_full - U_empty), 1e-6)
        
        return {
            "phi": phi,
            "psi": psi,
            "top_idx": top_idx,
            "conservation_residual": conservation_residual,
            "N_out": N_out,
            "N_in": N_in,
            "U_full": U_full,
            "U_empty": U_empty,
        }
```

### 6.4 `estimator/kernel_shap.py` — Outer game

```python
# modality_credit/estimator/kernel_shap.py
"""
Kernel SHAP for outer game over K items.

Reference: Lundberg & Lee 2017 / SHAP library KernelExplainer.
We use a custom implementation because SHAP lib's evaluator interface
is awkward for our utility function.
"""
import numpy as np
from scipy.special import comb

def kernel_shap_outer(utility, query_inst, N: int, seed: int = 42, seed_attention=False):
    K = len(query_inst.memory)
    rng = np.random.default_rng(seed)
    
    # Sample N coalition subsets
    if seed_attention:
        coalitions = _attention_seeded_coalitions(query_inst, N, rng)
    else:
        coalitions = _uniform_coalitions(K, N, rng)
    
    # Evaluate u for each coalition
    u_vals = np.zeros(N)
    for i, c in enumerate(coalitions):
        u_vals[i] = utility.evaluate(query_inst, c.tolist())
    
    # Solve weighted least squares: phi = argmin Σ w_S (u(S) - u(∅) - Σ_k S_k phi_k)^2
    # with constraint Σ phi_k = u(N) - u(∅)
    phi = _solve_wls(coalitions, u_vals, K, query_inst, utility)
    return phi

def _uniform_coalitions(K, N, rng):
    """Sample coalitions with SHAP kernel weights π(|S|) = (K-1) / (C(K, |S|) * |S| * (K-|S|))."""
    sizes = np.arange(1, K)  # exclude empty and full (deterministic)
    weights = (K - 1) / (comb(K, sizes) * sizes * (K - sizes))
    weights /= weights.sum()
    
    out = []
    for _ in range(N):
        s = rng.choice(sizes, p=weights)
        subset = np.zeros(K, dtype=bool)
        subset[rng.choice(K, size=int(s), replace=False)] = True
        out.append(subset)
    return np.array(out)

def _solve_wls(coalitions, u_vals, K, query_inst, utility):
    u_empty = utility.evaluate(query_inst, [False]*K)
    u_full = utility.evaluate(query_inst, [True]*K)
    
    X = coalitions.astype(float)
    y = u_vals - u_empty
    
    # Constrained least squares: Σ phi = u_full - u_empty
    # Solve via Lagrangian / projection
    phi, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    # Project to satisfy efficiency
    correction = (u_full - u_empty - phi.sum()) / K
    phi += correction
    return phi
```

### 6.5 `estimator/mc_permutation.py` — Inner game

```python
# modality_credit/estimator/mc_permutation.py
"""
Monte Carlo permutation Shapley for inner game (modalities within one item),
conditional on all other items frozen.
"""
import numpy as np

def mc_perm_inner(utility, query_inst, item_idx: int, N: int, seed: int = 42):
    K = len(query_inst.memory)
    item = query_inst.memory[item_idx]
    modalities = list(item.modalities.keys())   # e.g., ["vision", "text", "audio"]
    L = len(modalities)
    
    rng = np.random.default_rng(seed)
    psi = np.zeros(L)
    
    # Freeze all other items (all True)
    all_items_on = [True] * K
    
    for _ in range(N):
        perm = rng.permutation(L)
        current_modality_mask = {m: False for m in modalities}
        modality_masks = [dict.fromkeys(item.modalities.keys(), True) for _ in range(K)]
        modality_masks[item_idx] = current_modality_mask.copy()
        
        u_prev = utility.evaluate(query_inst, all_items_on, modality_masks)
        
        for ell_idx in perm:
            ell = modalities[ell_idx]
            modality_masks[item_idx][ell] = True
            u_curr = utility.evaluate(query_inst, all_items_on, modality_masks)
            psi[ell_idx] += (u_curr - u_prev) / N
            u_prev = u_curr
    
    return psi
```

### 6.6 `estimator/allocation.py` — Algorithm 1

```python
# modality_credit/estimator/allocation.py
"""
Closed-form sample allocation per Algorithm 1.

Variance per game: Var(φ̂) ∝ 2^K / N_out,  Var(ψ̂^(k)) ∝ 2^{L_k} / N_in^(k)
Budget: B = N_out + Σ N_in^(k)

Lagrangian solution:  N_out* ∝ √(2^K),  N_in^(k)* ∝ √(2^{L_k})
"""
import numpy as np

def allocate_budget(K: int, L_ks: list[int], B: int):
    """
    Args:
        K:     number of items
        L_ks:  list of within-item player counts
        B:     total utility-query budget
    Returns:
        N_out: int
        N_in:  list of int, length K
    """
    sqrt_outer = np.sqrt(2**K)
    sqrt_inners = np.sqrt(np.array([2**Lk for Lk in L_ks], dtype=float))
    Z = sqrt_outer + sqrt_inners.sum()
    
    N_out = int(np.floor(B * sqrt_outer / Z))
    N_in = [int(np.floor(B * s / Z)) for s in sqrt_inners]
    
    # Ensure at least 10 per game for variance stability
    N_out = max(N_out, 10)
    N_in = [max(n, 10) for n in N_in]
    return N_out, N_in
```

### 6.7 `pruning.py` — Modality-pruned retrieval

```python
# modality_credit/pruning.py
def modality_prune(query_inst, phi, psi, tau_1, tau_2):
    """
    M̃(Q) = { v_k^(ℓ) : φ_k > τ_1  ∧  ψ_{k,ℓ} > τ_2 }
    
    Returns: pruned context string + retention ratio
    """
    K = len(query_inst.memory)
    item_mask = [phi[k] > tau_1 for k in range(K)]
    modality_masks = []
    total_patches = 0
    kept_patches = 0
    
    for k, item in enumerate(query_inst.memory):
        mm = {}
        for ell_idx, m in enumerate(item.modalities.keys()):
            keep = (item_mask[k] and psi[k, ell_idx] > tau_2)
            mm[m] = keep
            total_patches += 1
            kept_patches += int(keep)
        modality_masks.append(mm)
    
    from modality_credit.masking import apply_masks
    pruned_context = apply_masks(query_inst.memory, item_mask, modality_masks)
    retention = kept_patches / total_patches
    return pruned_context, retention
```

---

## 7. Pre-pilot Audits（5/13 必做）

这是 plan v2 + 风险讨论后**最重要的新增工作**——在跑 Phase 1 toy pilot 之前，先验证三个底层假设是否成立。否则后续实验全失真。

### 7.1 Modality Leakage Audit

**目标**：验证我们的 mask 真的让 generator 看不到原信息。

**怎么测**：

```python
# scripts/01_pre_pilot_audit.py 节选
def audit_modality_leakage(utility, samples):
    """For each sample, test single-modality utility."""
    rows = []
    for inst in samples:
        K = len(inst.memory)
        for k in range(K):
            for keep_mod in ["vision", "text", "audio"]:
                modality_masks = [dict.fromkeys(m.modalities, True) for m in inst.memory]
                modality_masks[k] = {m: (m == keep_mod) for m in inst.memory[k].modalities}
                item_mask = [i == k for i in range(K)]
                u = utility.evaluate(inst, item_mask, modality_masks)
                rows.append({"sample_id": inst.query[:30], "item": k, "modality": keep_mod, "u": u})
    return pd.DataFrame(rows)
```

**Sanity checks**:

| 检查 | 期望 | 不通过怎么办 |
|---|---|---|
| 三个单模 utility 加起来 vs 全模 utility 的比值 | ~ 1.5–2.5× (说明 modality 有冗余但不完全冗余) | < 1.2 说明模态高度冗余, attribution 空间小; > 3 说明可能有 leak |
| Vision 单模 utility | ≥ 0.2 (vision 有信息) | < 0.05 说明 vision mask 后还有信息泄漏到 text |
| Text 单模 utility | ≥ 0.2 | 同上 |
| 全 mask (U(∅)) | ≤ 0.3 | > 0.3 见 7.2 |

**100 sample / ~0.3 GPU-h**。

### 7.2 U(∅) Baseline Measurement

```python
def audit_u_empty(utility, samples):
    u_empty_dist = [utility.evaluate(s, [False]*len(s.memory)) for s in samples]
    return {
        "mean": np.mean(u_empty_dist),
        "p25": np.percentile(u_empty_dist, 25),
        "p75": np.percentile(u_empty_dist, 75),
        "p_high": sum(u > 0.5 for u in u_empty_dist) / len(samples),
    }
```

**决策**:
- U(∅) mean < 0.2 → conservation 直接成立, 无需 normalize
- 0.2 < mean < 0.4 → **必须 normalize**: u_norm(S) = (U(S) − U(∅)) / (U(full) − U(∅))
- mean > 0.4 → query-only baseline 太强, M3-Bench 的 query 设计有问题; 换 benchmark 或筛 sample

### 7.3 Positional Invariance Check

```python
def audit_positional(utility, samples, n_shuffles=5):
    """For each sample, shuffle item order n times, measure U variance."""
    rows = []
    for inst in samples:
        K = len(inst.memory)
        u_vals = []
        for _ in range(n_shuffles):
            perm = np.random.permutation(K)
            shuffled = QueryInstance(
                query=inst.query, 
                memory=[inst.memory[i] for i in perm],
                gold_answer=inst.gold_answer,
            )
            u_vals.append(utility.evaluate(shuffled, [True]*K))
        rows.append({"std": np.std(u_vals), "mean": np.mean(u_vals)})
    return pd.DataFrame(rows)
```

**Sanity check**: position 引起的 U std / U mean 比值 < 5% → permutation invariance 实践上 OK
                  > 10% → 必须在 Owen 估计时显式 randomize position

### 7.4 M3-Agent Demo Verification

```bash
# scripts/00_setup_m3_agent.sh
cd vendor/m3_agent
pip install -e .
python demo.py --benchmark m3_bench --split val --n 5  # run their demo
# Verify the printed accuracy matches paper claim (Table 2 in M3-Agent paper)
```

如果跑不通或数字对不上 → 找 ByteDance 团队 issue / 改用 WorldMM 做主 codebase。

### 7.5 Audit 总报告

跑完 7.1–7.4 后产出 `results/phase0_audit/AUDIT_REPORT.md`，含 4 个表格 + 一段 verdict：

```
VERDICT: 
  - Modality leakage:    PASS / WARN / FAIL
  - U(∅) baseline:       <0.3 → no normalization needed
  - Positional invariance: PASS (std < 5%)
  - M3-Agent demo:       PASS (acc matches paper ±2%)
  
DECISION: PROCEED to Phase 1 toy pilot.
```

任何一项 FAIL → 当天解决或调整 setup，不进 Phase 1。

---

## 8. Phase-by-phase 实验脚本

### 8.1 Phase 1 — Toy Pilot (5/14, 0.5 GPU-h)

```python
# scripts/02_toy_pilot.py
@hydra.main(config_path="../configs", config_name="phase1_toy_pilot")
def main(cfg):
    utility = build_utility(cfg.model_name)
    samples = load_m3_bench(split="val", n=30, K=cfg.K)
    estimator = OwenEstimator(utility, top_n_inner=2, seed=cfg.seed)
    
    results = []
    for inst in samples:
        out = estimator.estimate(inst, budget_B=cfg.budget_B)
        results.append({**out, "query": inst.query})
    
    # Sanity checks
    var_phi = np.var([r["phi"] for r in results], axis=0).mean()
    theoretical_bound = (2**cfg.K) / (4 * cfg.budget_B)
    
    report = {
        "var_phi_empirical": var_phi,
        "var_phi_bound": theoretical_bound,
        "var_ratio": var_phi / theoretical_bound,
        "mean_conservation_residual": np.mean([r["conservation_residual"] for r in results]),
        "seed_reproducibility_spearman": _compute_spearman_across_seeds(samples, estimator),
    }
    
    # 3 decision tests
    decide_go_or_pivot(report)
    wandb.log(report)
```

**通过条件**: 见 plan v2 §决策表。

### 8.2 Phase 2 — Detection + Theory Validation (5/15, 2.2 GPU-h)

```python
# scripts/03_full_detection.py
def main(cfg):
    utility = build_utility(cfg.model_name)
    samples = load_m3_bench(split="test", n=100, K=cfg.K)
    estimator = OwenEstimator(utility, top_n_inner=2)
    
    # Run Owen attribution on all samples
    attrs = [estimator.estimate(s, budget_B=500) for s in samples]
    
    # Claim 1: top-vs-bottom quartile prune
    delta_acc = evaluate_top_bottom_quartile(utility, samples, attrs)
    
    # Baselines
    baseline_results = {}
    for name, fn in BASELINES.items():
        baseline_attrs = [fn(utility, s) for s in samples]
        baseline_results[name] = evaluate_top_bottom_quartile(utility, samples, baseline_attrs)
    
    # Claim 1.1: seeding ablation
    seeded = [estimator.estimate(s, budget_B=500, seed_attention=True) for s in samples]
    spearman_seeding = avg_spearman([a["phi"] for a in attrs], [s["phi"] for s in seeded])
    
    # Claim 1.6: real-data LOO variance
    loo_var = sweep_n_samples(utility, samples, N_grid=[50, 100, 200], n_seeds=5)
    
    save_results(cfg.output_dir, attrs, baseline_results, spearman_seeding, loo_var)
```

Claim 1.5（synthetic）单独跑：

```python
# scripts/03b_synthetic_scaling.py
def main(cfg):
    for K in [2, 4, 6, 8]:
        for L_k in [1, 2, 4, 6]:
            samples = generate_synthetic(K=K, L_k=L_k, n=20, gt_decisive_modality=True)
            attrs = [estimator.estimate(s, budget_B=cfg.budget_B) for s in samples]
            
            # Validate Cor 2c
            var_ratio_empirical = compute_var_ratio(attrs, samples)
            var_ratio_theoretical = (K * 2**L_k) / 2**(K * L_k)
            
            # Validate Alg. 1
            n_out_empirical, n_in_empirical = empirical_optimal_allocation(samples, cfg.budget_B)
            n_out_alg1 = (2**K)**0.5
            n_in_alg1 = (2**L_k)**0.5
            
            wandb.log({
                f"K{K}_L{L_k}/var_ratio_empirical": var_ratio_empirical,
                f"K{K}_L{L_k}/var_ratio_theoretical": var_ratio_theoretical,
                f"K{K}_L{L_k}/alloc_ratio_empirical": n_out_empirical / np.mean(n_in_empirical),
                f"K{K}_L{L_k}/alloc_ratio_alg1": n_out_alg1 / n_in_alg1,
            })
```

### 8.3 Phase 3 — Modality-Pruned Retrieval + Cross-Arch (5/16, 4.5 GPU-h)

```python
# scripts/04_pruning_pareto.py
def main(cfg):
    utility = build_utility(cfg.model_name)
    samples = load_m3_bench(split="test", n=100, K=cfg.K)
    estimator = OwenEstimator(utility)
    attrs = [estimator.estimate(s, budget_B=500) for s in samples]
    
    pareto = []
    for tau_1, tau_2 in [(.05, .05), (.1, .1), (.2, .2), (.3, .3)]:
        # Ours
        acc_ours, retention_ours = evaluate_pruning(samples, attrs, tau_1, tau_2)
        # Baselines
        acc_random = evaluate_random_drop(samples, retention_ours)        # critical control
        acc_item_only = evaluate_item_only_prune(samples, attrs, tau_1)
        acc_mma = evaluate_mma_baseline(samples)                          # MMA reliability score
        
        pareto.append({
            "tau_1": tau_1, "tau_2": tau_2, "retention": retention_ours,
            "acc_ours": acc_ours, "acc_random": acc_random,
            "acc_item_only": acc_item_only, "acc_mma": acc_mma,
        })
    
    save_pareto(cfg.output_dir, pareto)
    plot_pareto(pareto, save="figures/pareto.pdf")

# scripts/05_cross_architecture.py
def main(cfg):
    for arch in ["m3_agent_qwen", "worldmm_internvl", "mma_llava"]:
        utility = build_utility(arch)
        samples = load_arch_specific_benchmark(arch, n=50)
        attrs = [OwenEstimator(utility).estimate(s) for s in samples]
        pareto = evaluate_pareto(samples, attrs, tau_grid)
        save_pareto(f"results/phase3/cross_arch_{arch}.json", pareto)
```

### 8.4 Phase 4 — Hardening Pack (5/19+, 13 GPU-h)

#### Phase 4A — Memory Poisoning Detection (3 GPU-h)

```python
# scripts/06_poisoning_detection.py
def main(cfg):
    samples = load_m3_bench(split="test", n=100)
    
    poisoned_samples = []
    for s in samples:
        for atk in ["caption_flip", "audio_noise", "frame_swap", "entity_swap"]:
            ps = apply_adversarial(s, attack=atk, target_item_idx=0)  # always poison item 0
            poisoned_samples.append((ps, atk, 0))
    
    estimator = OwenEstimator(utility)
    
    detection_results = []
    for ps, atk, gt_poisoned_idx in poisoned_samples:
        attr = estimator.estimate(ps)
        # Detection signal: ψ across modalities of item 0 高度不一致
        inconsistency = compute_modality_inconsistency(attr["psi"][gt_poisoned_idx])
        phi_outlier = compute_phi_outlier_score(attr["phi"], gt_poisoned_idx)
        signal = inconsistency + phi_outlier  # combine
        detection_results.append({
            "attack": atk, "signal": signal, "label": 1,
        })
    
    # Negative samples
    for s in samples:
        attr = estimator.estimate(s)
        for k in range(len(s.memory)):
            signal = compute_modality_inconsistency(attr["psi"][k]) + compute_phi_outlier_score(attr["phi"], k)
            detection_results.append({"attack": "clean", "signal": signal, "label": 0})
    
    auroc = sklearn.metrics.roc_auc_score(
        [r["label"] for r in detection_results],
        [r["signal"] for r in detection_results],
    )
    
    # Baselines: random, retrieval similarity, perplexity, attention rollout
    baselines_auroc = {b: evaluate_baseline_detection(b, ...) for b in BASELINES}
    
    wandb.log({"auroc_ours": auroc, **baselines_auroc})
```

#### Phase 4B — LongVideoBench K=8 (10 GPU-h)

```python
# scripts/07_longvideobench_k8.py
def main(cfg):
    for K in [2, 4, 6, 8]:
        samples = load_longvideobench(K=K, n=300 if K == 8 else 50)
        utility = build_utility(cfg.model_name)
        estimator = OwenEstimator(utility)
        
        # Wall-clock time vs flat MM-SHAP
        t_hier_start = time.time()
        attrs = [estimator.estimate(s, budget_B=500) for s in samples[:10]]
        t_hier = (time.time() - t_hier_start) / 10
        
        if K <= 4:  # flat 在 K > 4 跑不动
            t_flat = time.time()
            flat_attrs = [flat_token_shapley(utility, s, budget_B=10000) for s in samples[:10]]
            t_flat = (time.time() - t_flat) / 10
        else:
            t_flat = np.inf
        
        # Claim 2 pruning on K=8
        pareto = evaluate_pruning(samples, attrs, tau_grid)
        
        wandb.log({
            f"K{K}/t_hier_per_query": t_hier,
            f"K{K}/t_flat_per_query": t_flat,
            f"K{K}/wedge_realized": t_flat / t_hier,
            f"K{K}/pareto": pareto,
        })
```

#### Phase 4C — Matching Lower Bound (0 GPU-h, all math)

5/19–5/21 写 LaTeX proof，无 GPU 工作。技术路径：

1. 把 Mann-Wright (1991) 关于 Shapley estimator 的 variance lower bound 重述
2. 将 single-game 推广到 two-level Owen game
3. 证明 Var ≥ c · 2^K/N + d · Σ 2^{L_k}/N for some constants c, d > 0
4. 这个下界和我们 Prop 2 的上界 在 constant factor 内匹配 → tight

#### Phase 4D — Failure Mode Taxonomy (0 GPU-h, re-analyze)

```python
# notebooks/04_failure_taxonomy.ipynb
def cluster_failures(samples, attrs):
    # Take all samples where U_full = 0 (model got it wrong)
    failed = [(s, a) for s, a in zip(samples, attrs) if a["U_full"] == 0]
    
    # Feature vector per failure: (φ_max, φ_entropy, ψ_max, ψ_entropy, modality_dominance)
    features = np.array([extract_failure_features(s, a) for s, a in failed])
    
    # Cluster
    from sklearn.cluster import KMeans
    labels = KMeans(n_clusters=5).fit_predict(features)
    
    # Interpret clusters: which modality dominates? are φ entropy high (no clear culprit)?
    interpret_clusters(features, labels, failed)
```

#### Phase 4E — Human Study (0 GPU-h, 7 days for annotation)

```python
# scripts/08_human_study_prep.py
def main(cfg):
    samples = load_m3_bench_failures(n=20)
    attrs = [OwenEstimator(utility).estimate(s) for s in samples]
    
    # Generate two versions of each trace: with attribution overlay, without
    for s, a in zip(samples, attrs):
        render_trace_html(s, a, save=f"human_study/with_attr/{s.id}.html")
        render_trace_html(s, attr=None, save=f"human_study/without_attr/{s.id}.html")
    
    # Annotator interface (Streamlit)
    launch_streamlit_app(
        traces_with=f"human_study/with_attr/",
        traces_without=f"human_study/without_attr/",
        n_annotators=20,
    )
```

**Annotator UI design**:
- 每页一个 trace: query, memory items, agent answer, gold answer
- 问题: "Which (item, modality) is most responsible for the wrong answer?" (选择题, K × max(L_k) 个选项)
- 计时: time from page load to submission
- 一半 annotator 看到 attribution overlay (highlighted φ, ψ in heatmap), 一半看不到
- Ground truth: 我们手工标 20 个 trace 的 "真正原因"

**Metrics**: accuracy of diagnosis (vs GT), median time per case, statistical test (t-test) for difference between two groups.

---

## 9. 数据 Pipeline

### 9.1 M3-Bench

```python
# modality_credit/data/m3_bench.py
def load_m3_bench(split="val", n=None, K=4):
    """
    Load M3-Bench (Q, M, y*) tuples.
    
    M3-Agent's repo has a `data/m3_bench.json` with annotated QA pairs.
    For each QA, the M3-Agent retrieval module pre-stores top-K retrieved items.
    We load both and assemble QueryInstance.
    """
    raw = json.load(open("vendor/m3_agent/data/m3_bench.json"))
    items_db = load_items_db()  # episodic memory store
    
    out = []
    for entry in raw[split]:
        if n and len(out) >= n:
            break
        retrieved_ids = entry["retrieved_top_k"][:K]
        memory = [build_memory_item(items_db[i]) for i in retrieved_ids]
        out.append(QueryInstance(
            query=entry["question"],
            memory=memory,
            gold_answer=entry["answer"],
        ))
    return out

def build_memory_item(raw):
    return MemoryItem(
        item_id=raw["id"],
        modalities={
            "vision": load_frame(raw["frame_path"]),
            "text": raw["caption"],
            "audio": raw["audio_transcript"],
            "scene": raw.get("scene_meta", ""),
        },
        metadata={"timestamp": raw["t"], "source_confidence": raw.get("conf", 1.0)},
    )
```

### 9.2 Synthetic Generator (Claim 1.5)

```python
# data/synthetic/generator.py
def generate_synthetic(K, L_k, n, gt_decisive_modality=True):
    """
    Generate synthetic (Q, M, y*) tuples with injected ground-truth decisive modality.
    
    Recipe:
    1. Sample a "scene" (e.g., kitchen, office)
    2. Generate K memory items with random vision/text/audio
    3. Pick one decisive (item k*, modality ℓ*) and inject the answer-relevant signal
    4. Verify: removing the decisive modality makes accuracy drop ≥30pp
    """
    samples = []
    for _ in range(n):
        scene = random.choice(SCENES)
        k_star = random.randint(0, K-1)
        l_star = random.choice(["vision", "text", "audio"][:L_k])
        
        memory = [random_memory_item(scene) for _ in range(K)]
        memory[k_star] = inject_decisive_signal(memory[k_star], modality=l_star)
        
        q, y = make_qa_for_decisive(memory[k_star], modality=l_star)
        samples.append(QueryInstance(q, memory, y), gt_idx=(k_star, l_star))
    return samples
```

### 9.3 Adversarial Generator (Phase 4A)

```python
# modality_credit/adversarial/__init__.py
def apply_adversarial(query_inst, attack: str, target_item_idx: int):
    """
    Inject adversarial content into target_item_idx.
    
    Attacks:
    - caption_flip:   replace caption with semantically opposite text
    - audio_noise:    replace transcript with random unrelated text
    - frame_swap:     replace frame with a random unrelated frame
    - entity_swap:    swap one entity name throughout the item (e.g., "knife" -> "spoon")
    """
    poisoned = copy.deepcopy(query_inst)
    target = poisoned.memory[target_item_idx]
    
    if attack == "caption_flip":
        target.modalities["text"] = flip_caption(target.modalities["text"])
    elif attack == "audio_noise":
        target.modalities["audio"] = sample_random_transcript()
    elif attack == "frame_swap":
        target.modalities["vision"] = sample_random_frame()
    elif attack == "entity_swap":
        target.modalities = entity_swap_all_modalities(target.modalities)
    
    return poisoned
```

---

## 10. Baselines 实现

| Baseline | File | Cost vs ours |
|---|---|---|
| Flat token Shapley | `baselines/flat_shapley.py` | 800× expensive |
| Item-only Shapley | `baselines/item_only.py` | 0.5× cheaper |
| Attention rollout | `baselines/attention_rollout.py` | 0.01× (1 forward) |
| LLM self-report | `baselines/llm_self_report.py` | 1× (1 prompt) |
| MMA reliability | `baselines/mma_reliability.py` | 0.01× (no LLM) |
| Random drop | `baselines/random_drop.py` | (just for Claim 2 control) |

```python
# baselines/attention_rollout.py
def attention_rollout(utility, query_inst):
    """Use Qwen-VL last-layer attention as saliency, aggregate per (item, modality)."""
    model = utility.generator.model
    # ... extract attention, aggregate per token-block, normalize to a [0,1] score per modality
    return phi_attn, psi_attn

# baselines/llm_self_report.py
def llm_self_report(utility, query_inst):
    """Prompt the generator: 'Which memory item × modality did you rely on?'"""
    prompt = build_self_report_prompt(query_inst)
    response = utility.generator.generate_text(prompt)
    return parse_self_report(response)  # → (phi, psi) on score scale
```

---

## 11. Adversarial Generator (Phase 4A)

详见 §9.3. 4 类攻击的 implementation 细节，每类要：

1. 不破坏 item 的整体结构（仍是合法的 (vision, text, audio) 组合）
2. 改动局限在一个 item 一个 modality（这样 ground truth idx 明确）
3. 改动足够"明显"——半数 case generator 答错

**Sanity test**: 在 100 个 clean sample 上跑 vs 100 个 poisoned，poisoned 平均 accuracy 应 ≤ 50%（即 attack 真的有效）。

---

## 12. Logging & WandB

### 12.1 WandB project structure

```
modality_credit/                     # project name
  ├── phase0_audit/
  ├── phase1_pilot/
  ├── phase2_detection/
  ├── phase3_pruning/
  └── phase4_*/
```

### 12.2 关键 metric

```python
# Phase 1
wandb.log({
    "phase1/var_phi_empirical": ...,
    "phase1/var_phi_bound": ...,
    "phase1/var_ratio": ...,
    "phase1/conservation_residual_mean": ...,
    "phase1/spearman_5seeds": ...,
    "phase1/u_empty_mean": ...,
})

# Phase 2
wandb.log({
    "phase2/delta_acc_ours": ...,
    "phase2/delta_acc_flat": ...,
    "phase2/delta_acc_item_only": ...,
    "phase2/delta_acc_attention": ...,
    "phase2/delta_acc_self_report": ...,
    "phase2/seeding_spearman": ...,
})

# Phase 3
wandb.log({
    "phase3/pareto_curve": wandb.plot.line_series(...),
    "phase3/delta_acc_at_40_retention/ours": ...,
    "phase3/delta_acc_at_40_retention/random": ...,
    "phase3/cross_arch_std": ...,
})
```

### 12.3 Sidecar 数据

attribution tensor 太大不进 WandB。存到 disk：

```
results/phase2_detection/
  ├── attrs_qwen_vl_7b.npz      # all 100 (phi, psi) arrays
  ├── baselines_attrs.npz
  ├── metadata.json
  └── pareto.csv
```

---

## 13. Compute Deployment

### 13.1 Modal serverless (推荐)

```python
# modal_deploy.py
import modal

app = modal.App("modality-credit")
image = modal.Image.debian_slim().pip_install_from_pyproject("pyproject.toml")

@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    secrets=[modal.Secret.from_name("hf_token")],
)
def run_phase(phase: str, cfg_path: str):
    import subprocess
    subprocess.run(["python", f"scripts/{phase}.py", "--config-path", cfg_path], check=True)

@app.local_entrypoint()
def main(phase: str = "02_toy_pilot"):
    run_phase.remote(phase, f"configs/{phase}.yaml")
```

`modal run modal_deploy.py --phase 02_toy_pilot`

### 13.2 Vast.ai backup

```bash
# rent on-demand A100
vastai create instance <offer-id> --image pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime --disk 100 --ssh
ssh root@<ip> "cd /workspace && git clone ... && bash scripts/00_setup_m3_agent.sh && python scripts/02_toy_pilot.py"
```

### 13.3 Cost estimate

| Service | Phase | Hours | Rate | Cost |
|---|---|---|---|---|
| Modal A100-80GB | Phase 0–3 | 8 | $4/h | $32 |
| Modal A100-80GB | Phase 4A | 3 | $4 | $12 |
| Modal A100-80GB | Phase 4B (long) | 10 | $4 | $40 |
| Modal H100 (optional, for 72B model ablation) | optional | 5 | $8 | $40 |
| **Total core** | | | | **~$84** |
| **Total + 30% buffer** | | | | **~$110** |

---

## 14. Risk Monitoring & Decision Tree

### 14.1 Phase 1 决策

```
toy pilot 跑完 →
  ├── 全部 PASS (3 项)              → 进 Phase 2
  ├── 1 项 WARN                    → fix 当晚, 重跑
  ├── 1 项 FAIL                    → 检查实现 bug, 重跑
  └── 全部 FAIL                    → **pivot to backup direction** (CMA for write-policy)
```

### 14.2 Phase 3 venue tier 决策

```
Claim 2 ΔAcc at 40% retention →
  ├── ≥ 10pp                       → 顶会路径 (NeurIPS D&B / CVPR full paper)
  ├── 5–10pp                       → 次顶 (ICLR / EMNLP short)
  └── < 5pp                        → 降为 case study, paper 仅靠 Claim 1+1.5+1.6 + Phase 4
```

### 14.3 自动告警

```python
# 在 each Phase 末加
def phase_alert(phase, results):
    if phase == "phase1" and results["var_ratio"] > 3:
        send_feishu_alert("🚨 Phase 1 Failed: var ratio > 3×, consider pivot to backup")
    if phase == "phase3" and results["delta_acc_at_40"] < 5:
        send_feishu_alert("⚠️ Phase 3: ΔAcc < 5pp, prepare downgrade narrative")
```

---

## 15. 实施时间线

| Day | Date | Tasks | Cost |
|---|---|---|---|
| 1 | 5/12 (今天) | 写完此 framework doc + advisor sync 准备材料 | 0 |
| 2 | 5/13 | M3-Agent 装 + repo scaffold + 写完 utility.py / masking.py / estimator/* | 0 |
| 3 | 5/14 morning | 跑 Pre-pilot audits (7.1-7.4), 4 个表格出来 | 0.3 GPU-h |
| 3 | 5/14 afternoon | Phase 1 toy pilot (30 QA) + 决策 | 0.5 GPU-h |
| 4 | 5/15 | Phase 2: Claim 1 + 1.1 + 1.5 + 1.6 | 2.2 GPU-h |
| 5 | 5/16 | Phase 3: Claim 2 Pareto + Claim 3 cross-arch | 4.5 GPU-h |
| 6 | 5/17 | 1-pager: figures (Pareto curve, variance-ratio plot) + 文字 | 0 |
| 7 | 5/18 | Advisor sync | 0 |
| 8–14 | 5/19–5/25 | Phase 4A (poisoning) + 4C (lower bound proof) | 3 GPU-h |
| 15–17 | 5/26–5/28 | Phase 4B (LongVideoBench K=8) | 10 GPU-h |
| 18–21 | 5/29–6/01 | Phase 4D (taxonomy) + 4E (human study annotation collection) | 0 |
| 22+ | 6/02 起 | Paper writing | — |
| ~7/15 | NeurIPS 2026 deadline (estimated) | — | — |

---

## 16. 相关文档

- **Method plan (主)**: [`multimodal_memory_credit_method_plan_repositioned.md`](multimodal_memory_credit_method_plan_repositioned.md)
- **Brief proposal (1-pager)**: [`multimodal_memory_credit_proposal_brief.md`](multimodal_memory_credit_proposal_brief.md)
- **Novelty check report**: [`multimodal_memory_credit_novelty_report.md`](multimodal_memory_credit_novelty_report.md)
- **Idea report (上游)**: [`multimodal_memory_credit_idea_report.md`](multimodal_memory_credit_idea_report.md)
- **HTML 整合版**: [`multimodal_memory_credit_proposal.html`](multimodal_memory_credit_proposal.html)

---

## Appendix A — 检查清单 (5/13 implementation 启动前过一遍)

- [ ] M3-Agent vendor submodule 克隆成功，demo 能复现 paper Table 2 数字
- [ ] WorldMM / MMA submodule 也克隆好（Phase 3 跨架构用）
- [ ] Qwen-2.5-VL-7B weights 下载（HuggingFace, ~17GB）
- [ ] WandB / Modal account + secrets 配好
- [ ] `utility.py` 通过 unit test（feed 假数据返回 0/1）
- [ ] `masking.py` 通过 leakage audit smoke test（≥ 5 sample）
- [ ] `estimator/owen.py` 在 K=2, L_k=2 toy case 上和 brute-force Shapley 结果一致（±5%）
- [ ] `allocation.py` 在 K=4, L_k=4, B=500 case 上输出 N_out ≈ 94, N_in ≈ 51
- [ ] 跑 `pytest tests/ -v` 全绿
- [ ] `scripts/01_pre_pilot_audit.py` 跑通（不看结果，只确保 pipeline 完整）

跑过这 10 项后，进 5/14 真正的 audit + pilot。

---

## Appendix B — 常见坑 & FAQ

**Q: M3-Agent retrieval 不能直接调用怎么办？**
A: 直接读 M3-Agent 的 cached retrieval output（他们 paper Table 5 复现需要）；不用我们重跑 retrieval。

**Q: U(S) 用 exact match 太严苛怎么办？**
A: 切到 LLM-judge：用 GPT-4o-mini prompt "Is `{gen}` semantically equivalent to `{gold}`? Yes/No"。引入 noise 但 OK。

**Q: Kernel SHAP 输出有负值，conservation 怎么搞？**
A: 这是 KSHAP 的已知行为（regression-based fitting 可能输出负 φ）。Conservation 应该按 |Σ φ - (U(M)−U(∅))| 检查，而非 |Σ |φ| − ...|。负的 φ 表示该 item 对答对*有负面贡献*（include 反而拉低 accuracy），这是有意义的信号。

**Q: 跑 100 QA × 500 budget = 50K forward passes，A100 上多久？**
A: Qwen-2.5-VL-7B 单次 forward ~0.3s（context ~3K tokens）→ 50K × 0.3s ≈ 4.2 hours = 4.2 GPU-h。和 plan 估算一致。

**Q: Audit 发现 modality leakage 严重（vision 单模 acc > text 单模 + audio 单模）怎么办？**
A: 大概率是 caption 里编码了 vision 信息（"a person cuts a red fruit on a wooden cutting board"——caption 已经把视觉描述出来了）。Fix: 使用 stripped caption（删除 visual descriptors），或承认"我们的 'caption' modality 实际是 caption + vision summary 的复合"，paper limitation 写明。

**Q: Phase 4E 找不到 20 个 annotator 怎么办？**
A: 5 个 lab 同事 × 4 个 trace 每人也行，n=20 是 nice-to-have。或换 Amazon MTurk（成本 ~$50）。
