# Transformer Gradient Flow

> 由於寫完論文後作者這個人燃盡了 所以此次更新的 README 由 AI 更新 請見諒

## Paper (TAAI 2026)

**Which Direction Matters? A Paired Single-Path Ablation of Embedding Re-injection in Post-LN Transformers.**
Submitted to the TAAI 2026 high-school special session.

This section maps the paper to the code and data in this repository. The rest of the README (in Chinese)
documents the whole project; the paper corresponds to §5.4.

**Claim.** For a 16-layer Post-LN Transformer on a previous-token copy task, forward re-injection of
token content rescues training; the backward bypass speeds it up but is not necessary.

| Copy task, 15 paired seeds | Escapes (loss < 0.1) | McNemar *p* vs. Post-LN | Median final loss |
|---|---:|---:|---:|
| Post-LN | 0/15 | — | 4.1621 |
| + re-injection | 15/15 | 6.1e-5 | 0.0025 |
| + re-injection, backward bypass detached | 15/15 | 6.1e-5 | 0.0051 |
| + re-injection, MLP projection (AdaIN-local's conditional parameter count) | 15/15 | 6.1e-5 | 0.0025 |
| + re-injection of the token embedding only | 15/15 | 6.1e-5 | 0.0022 |
| + re-injection of the positional embedding only | 0/15 | 1 | 4.1621 |
| + re-injection of a frozen random per-position vector | 0/15 | 1 | 4.1621 |
| AdaIN-local (not a causal comparison: it also replaces LayerNorm with InstanceNorm) | 9/15 | 0.0039 | 0.0771 |

The detached version escapes later on every paired seed (by 176–294 steps; sign test *p* = 6.1e-5).
The rescue persists at lr = 1e-4 (15/15 vs. 0/15) but not on modk, which adds tokens 16 positions apart:
there re-injection matches Post-LN seed for seed, while Pre-LN makes partial progress on 12/15 seeds.
The evidence covers one task family, one depth, and one machine.

**Where each result lives**

| In the paper | Data | Produced or checked by |
|---|---|---|
| Table 1, Fig. 2, Results | `results/reinject_n15_paired/` | `make_table2.py`, `paper/make_figs.py` |
| Threats: lr = 1e-4 | `results/threats_lr1e-4/` | `verify_paper_numbers.py` |
| Threats: modk | `results/threats_modk/` | `verify_paper_numbers.py` |
| Threats: attention by layer | `results/layer_attn_n15/` | `analyze_layer_attn.py` |
| Pairing: 260 shared tensors and the step-0 logits are identical | — | `check_pairing.py` |
| Every number in the paper | all of the above | `verify_paper_numbers.py` |

**Check without retraining** (a CPU is enough):

```bash
pip install -r requirements.txt
python check_pairing.py          # every configuration PASS: identical to ln_post at step 0
python verify_paper_numbers.py   # recomputes every number in the paper; last line: ALL PASS
python make_table2.py            # regenerates Table 1 into results/reinject_n15_paired/table2.{md,tex}
```

**Retrain.** Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before starting Python, and pick a new `--out-dir`
so that the published results are not overwritten:

```bash
python run_experiment.py --task copy --seeds 15 --deterministic --out-dir paired_rerun --only ln_post adain_loc post_reinject post_reinject_detach post_reinject_mlp post_reinject_tok post_reinject_pos post_reinject_rand pre_ln deepnorm
python run_experiment.py --task copy --seeds 15 --deterministic --lr 1e-4 --out-dir lr1e-4_rerun --only ln_post post_reinject
python run_experiment.py --task modk --seeds 15 --deterministic --out-dir modk_rerun --only ln_post post_reinject pre_ln
```

**Setup.** 16 layers, d = 128, 4 heads, d_ff = 512 with GELU, no dropout; vocabulary 64, sequence length 64,
batch 32; Adam at 3e-4 without weight decay, 1,500 steps, no warmup, no learning-rate schedule, no gradient
clipping. Linear layers use Xavier-uniform weights and zero biases; token and position embeddings are drawn
from N(0, 0.02²). Seed *s* initializes the model with `torch.manual_seed(s)`, and the data come from a separate
generator seeded 1234 + *s*, so every configuration sees the same sequences. Each batch is freshly sampled,
so the training loss is also a held-out loss. An escape is a mean loss below 0.1 over the last 10 steps.
The results were produced with torch 2.8.0+cu128 on one RTX 5070 (Windows 11, Python 3.13.3).

**Notes on the data**

- Pairing needs the current `model.py` and `normalizations.py`. They were fixed on 2026-08-28 so that added
  modules no longer draw from the global random stream. Folders produced before the fix, such as
  `results/reinject_n15/` and `results/h3_n15/`, are not paired. With the current code, configurations that add
  conditional modules (AdaIN, SPADE, re-injection) start from different trunk weights than they did in those
  folders, so their per-seed numbers will not be reproduced exactly. Configurations without such modules are
  unaffected.
- `attn_to_target_*` is now the maximum over (layer, head) of the attention to t − 1. The older definition,
  which averages the heads within each layer, is kept as `*_headmean`. Folders produced before the change use
  the older definition.
- The new folders' `provenance.json` files record `git_commit: null`, because the runs were made outside a
  git checkout. `results/layer_attn_n15/` was rerun on 2026-09-16 with the final code and matches
  `reinject_n15_paired` seed for seed; `verify_paper_numbers.py` checks this.

---

> **專案目的**:探討同一個問題:**梯度在 Transformer 裡怎麼走、哪些設計會影響 trainability**。
> 本專案以 9 種 normalization × 4 種 placement 為切入點,另加 10 個診斷對照,共 25 個配置(主表報 15 個)。
>
> 這是一個**重現與視覺化型的實驗研究** —— 用嚴格控制變因的合成實驗,從 gradient flow 的角度
> 重現並視覺化權威論文(Xiong et al. 2020、Wang et al. 2022 等)的既有結論。
> 對移植自影像生成領域的 conditional normalization(AdaIN / SPADE),
> 本專案附加了探索性的對照量測(gradient decomposition、granularity ablation、純重注入對照)。
>
> 這算是一個嘗試做的一份研究實驗(?) 我不確定。
> 這份專案完成於我高中畢業後、升大學前的暑假,沒有指導。
>
> H1、H2 那部分不是原創研究。核心結論:Post-LN 的 gradient path 病灶、Pre-LN 的 identity shortcut、
> DeepNorm 的修復機制 —— 均已見於 Xiong et al. (2020) 與 Wang et al. (2022);
> 我做的是在受控的合成環境下重現並視覺化這些結論,用來確認自己是否真的理解了它們,
> 以及解決我自己的疑問。
>
> 帶一點原創性的是 H3 那條線:AdaIN-local 的 granularity ablation(15 seeds),
> 以及後續為了找出機制而做的純重注入對照。後者整理成了 TAAI 2026 的論文(見最上方 Paper 一節):
> 用配對種子與單一路徑消融,把一條繞過 normalization 的重注入路徑拆成正向與反向兩個方向,
> 結論是**救援來自正向重注入的 token 內容;反向梯度旁路會加速,但不是必要的**(§5.4)。
> 「加一條繞過 normalization 的路徑有用」本身不是新發現(B2T、DenseFormer、input injection 都屬這一類,§6.3);
> 據我們所知,新的是把兩個方向拆開來檢驗。
> 適用範圍:**單一任務家族 × 單一深度 × 單一機器**;lr = 1e-4 時仍成立,換成 modk 就不成立(§5.4 (e))。


**術語慣例**:AI 術語一律使用文獻中的英文原文,中文只用於敘述。
`normalization`(歸一化/正規化)、`placement`(擺放位置)、`gradient path`(梯度回傳通路)、
`identity shortcut`(恆等捷徑)、`bypass`(繞過主幹的旁路)、`granularity`(條件粒度)、
`trainability`(可訓練性)、`future leakage`(未來資訊洩漏)。

---

## 1. Question

**Normalization 影響 Transformer trainability 的機制,究竟是「改變特徵分布」,還是「改變梯度回傳路徑(gradient path)」?**

## 2. Background: Normalization and Gradient Paths

### 2.1 Backpropagation 是 Jacobian 的連乘

第 l 層收到的梯度是後面所有層 Jacobian 的連乘:

$$\frac{\partial L}{\partial h_l} = \frac{\partial L}{\partial h_N}\; J_N\, J_{N-1} \cdots J_{l+1}$$

每層 Jacobian 的尺度只要系統性地偏離 1 一點點,乘 2N 次(每個 Block 有 attention 與 FFN 兩個子層)之後就是指數級失衡。

### 2.2 LN 的 Jacobian(推導)

對單一 token 的特徵 $x \in \mathbb{R}^d$,無 affine 的 LayerNorm 為 $y = (x - \mu \mathbf{1})/\sigma$,其中 $\mu = \frac{1}{d}\mathbf{1}^\top x$、$\sigma^2 = \frac{1}{d}\|x-\mu\mathbf{1}\|^2 + \epsilon$。其 Jacobian 為:

$$\frac{\partial y}{\partial x} = \frac{1}{\sigma}\left(I - \frac{1}{d}\mathbf{1}\mathbf{1}^\top - \frac{1}{d}\,\hat{y}\hat{y}^\top\right),\qquad \hat{y} = \frac{x-\mu\mathbf{1}}{\sigma}$$

三個成分:**(a)** 整體縮放 $1/\sigma$;**(b)** 去均值投影 $I - \mathbf{1}\mathbf{1}^\top/d$;**(c)** 去半徑方向投影 $I - \hat{y}\hat{y}^\top/d$。投影項只削去兩個方向,對尺度影響微小;**主導梯度尺度的是 $1/\sigma$** —— 梯度反向穿過 LN 時,會被除以「前向 activation 的標準差」。

RMSNorm($y = x/\mathrm{rms}(x)$)的 Jacobian 是 $\frac{1}{r}(I - \hat{y}\hat{y}^\top/d)$:同樣的 $1/\sigma$ 結構,只少了去均值投影。**由此可預測:RMSNorm 與 LN 的 gradient flow 行為應幾乎相同**,而這個預測要在**兩者都會收斂**的 placement 下才驗得動(見 §5.3)。

### 2.3 Placement 決定 gradient path 的形狀

| Placement | 殘差主幹(residual stream)單層 Jacobian | 對 gradient flow 的含義 |
|---|---|---|
| **Post-LN**:`x = LN(x + F(x))` | $J_{\mathrm{LN}} \cdot (I + J_F)$ | $1/\sigma_l$ 坐在主幹上,連乘 2N 次;且 $\sigma_l$ 會隨訓練中 residual stream 的尺度漂移而變化 |
| **Pre-LN**:`x = x + F(LN(x))` | $I + J_F \cdot J_{\mathrm{LN}}$ | 恆等項保底:$\partial L/\partial h_l$ 至少包含 $\partial L/\partial h_N$ 的直達成分,不會指數衰減 |
| **Sandwich-LN**:`x = x + LN(F(LN(x)))` | $I + J_{\mathrm{LN}} J_F J_{\mathrm{LN}}$ | 保留 identity shortcut,並額外壓制分支輸出的前向尺度(CogView 用於抑制數值溢出) |
| **DeepNorm**:`x = LN(αx + F(x))`,$\alpha=(2N)^{1/4}$,V/O/FFN 初始權重乘 $\beta=(8N)^{-1/4}$ | $J_{\mathrm{LN}} \cdot (\alpha I + J_F)$ | 初始時 $J_F$ 被 β 壓小、LN 輸入由 αx 主導 ⇒ 整層 Jacobian 接近等距(isometry);DeepNet 論文的觀點是把單步模型更新量 bound 在常數級 |

> **這張表最直覺的讀法是錯的:$1/\sigma$ 在 Post placement 下不是「阻斷器」,是「穩定器」。**
> 把 Post-LN 的 $J_{\mathrm{LN}}$ 換成恆等(straight-through,前向逐元素不變),初始 block ratio
> 從 **1.82 爆到 3908**(層間放大率 1.057 → 1.793,而 $1.793^{15} \approx 4000$):殘差項
> $(I + J_F)$ 的範數 > 1、$J_{\mathrm{LN}}$ 的範數 < 1(主幹上實測 σ ≈ 1.12,又投影掉 2 個方向),
> 兩者相乘 ≈ 1.06 近乎等距,移掉 LN 的反向就等於移掉那個抑制項。
> 至少**在初始化時**是如此(訓練後主幹確實崩塌,見 §6.1 H2);這也是 §5.5 那條解耦路線走不通的原因。

### 2.4 Conditional Normalization:同時開了兩條路

AdaIN / SPADE 的 affine 參數 γ、β 由**條件輸入**經一個淺網路生成。本專案的條件是自產的:
`cond = x.clone()`,x 是 embedding 的輸出(`model.py`)。這一行同時開了兩條路,而**兩條都可能救活訓練**:

| | 路徑 | 機制 |
|---|---|---|
| **(a) 反向** | `cond.grad` 從每個 norm 點跳過中間所有 LN,直接回到 embedding | gradient bypass:梯度不必穿過主幹上的 LN 連乘,就能回到 embedding |
| **(b) 正向** | 每層的 γ/β 直接讀到乾淨的 per-position embedding | 從 embedding 拉到全部 16 層輸入的重注入,類似 DenseNet 式的 skip |

(b) 還有跨位置效果:第 l 層算出的 `h·(1+γ(cond_t)) + β(cond_t)` 會寫回殘差流,於是第 l+1 層的
attention 在位置 t 讀取位置 t−1 時,讀到的是「含有乾淨 embedding_{t−1} 成分」的向量。
**兩者要靠 `cond = x.clone().detach()` 的對照才分得開**(正向逐位元保留、反向完全切斷),
結果見 §5.4 —— 救援靠的是 (b) 帶進來的 **token 內容**;(a) 會加速,但不是必要的。
detach 切掉的只有 (a) 這條流回 embedding 的梯度,條件模組自己的權重照樣收到參數梯度、照樣在學。
兩個方向也不能完全分開:(b) 改變了之後每一個 LN 的輸入,也就改變了它們的 Jacobian,
detach 只移除顯式的旁路梯度(§7 第 10 點)。

## 3. Hypotheses

- **H1(主假設,可證偽版)**:在固定 placement 下更換 normalization 的統計維度
  (batch / sequence / channel),對 copy 任務 trainability 的影響**小於**更換 placement 的影響。
  *預測:9 種 normalization 全部落在同一個窄區間,而同一個 LN 換 placement 跨四個數量級。*
  *(現象成立;「機制是什麼」已排除一個候選但尚未回答,見 §6.1。)*
- **H2**:Post placement 的失敗不是初始化瞬間的病灶,而是**訓練過程中主幹梯度逐步崩塌**的動態過程。
  *預測:初始化時逐層梯度失衡溫和,訓練結束時劇烈。*
  *(現象成立且量得精確,但因果力已被直接干預否證,見 §5.5。)*
- **H3**:若 conditional normalization 能在 Post placement 下逃脫,關鍵變因是條件的
  **granularity**(per-position vs global)。
  *預測:以三方 ablation 孤立 —— AdaIN(global 向量)/ AdaIN-local(per-position,與 AdaIN
  唯一差別是不做 average pooling)/ SPADE(per-position + 共享 MLP)。*
  *(granularity 成立;逃脫機制由直接對照確定為正向重注入,而且要的是 token 內容、不是位置訊號,見 §5.4。)*
- **H4**:條件路徑只能在「位置內」調變,無法跨位置搬運資訊;解題所需的「取得前一格 token」
  仍必須由 attention 完成。
  *預測:成功收斂的模型,attention 對前一格的權重遠高於 uniform baseline ≈ 0.059。*

## 4. Method

### 4.1 三個任務:從 diagnostic probe 到真實資料

三個難度/真實度不同的任務,各跑完全相同的 15 配置 × 3 種子對照(輸出到 `results/<task>/`):

| 任務 | 定義 | 角色 |
|---|---|---|
| **copy** | `target[t] = x[t-1]`,token 均勻抽樣 | **Diagnostic probe**:答案只在前一格,per-position 的調變無法單獨解題,必經 attention;未來 token 與答案獨立 |
| **modk** | `target[t] = (x[t] + x[t-16]) mod 64` | **Long-range dependency**:答案同時需要本地的 x[t] 與 16 步前的 x[t-16],attention 必須跨 16 格搬運 |
| **char_lm** | Tiny Shakespeare 字元級 language modeling,`target[t] = x[t+1]` | **真實資料**:檢驗合成任務的結論是否在真實分佈上重現(90/10 連續切分,報 held-out validation) |

copy 任務的兩個關鍵性質(由 §4.4 的 attention probe 驗證,不只靠論證):
**(1)** 答案只存在於前一個位置 —— FFN 與 per-position modulation 都只能取用位置 t 自己的資訊,
跨位置搬運只有 attention 做得到,解題必然經過 attention;
**(2)** token 均勻獨立抽樣 ⇒ 未來 token 與答案統計獨立(對討論 non-causal normalization 的公平性
很重要,見 §7 第 2 點)。modk 同樣滿足這兩點;char_lm 的 target 在未來,對 IN/BN 家族有真實的
leakage,見 §5.2 第 2 點。

### 4.2 模型與控制變因

16 層、d_model=128、4 heads、d_ff=512、seq_len=64、batch=32。
**所有配置共用**:初始化種子、資料流、Adam(lr=3e-4)、訓練步數(1500)、**刻意不加 warmup**
(warmup 會遮住 Post-LN 的病灶)。主表 3 個隨機種子,報 mean±std(ddof=1);
H3 那條線與機制實驗為 15 種子,2×2 解耦為 10 種子。

其他共同設定:FFN 用 GELU、沒有 dropout;Linear 層 Xavier-uniform 初始化、bias 為 0;token 與 position
embedding 為 N(0, 0.02²);Adam 沒有 weight decay;沒有 gradient clipping,也沒有 learning rate schedule。
copy / modk 每一步都抽全新的隨機序列,所以 training loss 同時就是 held-out loss。
**「解掉」= 最後 10 步的平均 loss < 0.1。**

同一顆種子 s:模型用 `torch.manual_seed(s)` 初始化,資料由另一個 `Generator` 以 `1234 + s` 產生,
所以各配置看到的資料序列完全相同。**但 2026-08-28 以前的程式裡,多了條件模組(AdaIN / SPADE / 重注入)
的配置雖然用同一顆種子,主幹的初始權重並不相同**(模組建構時多抽了 RNG);修正與影響見 §5.4 (0)。

**參數量不相等 —— A 組不是 equal-parameter 比較**,conditional normalization 多了 γ/β 的生成層:

| 配置 | 參數量 | 相對 LN |
|---|---:|---:|
| BN / LN / IN / GN / DeepNorm | 3,196,992 | — |
| RMSNorm | 3,192,896 | −0.1% |
| SPADE | 3,985,472 | **+24.7%** |
| AdaIN / AdaIN-local | 4,245,568 | **+32.8%** |
| 純重注入 / E2 的三個變體(診斷對照,§5.4) | 3,725,376 | +16.5% |
| 純重注入(MLP)(診斷對照,§5.4) | 4,253,760 | +33.1% |

A 組的贏家 AdaIN-local 比 LN-Post 多 33% 參數,還多一條輸入重注入路徑。
**但 AdaIN vs AdaIN-local 是精確等參數的**(皆 4,245,568),γ/β zero-init 時兩者在數學上是
同一個模型(初始 gradient ratio 完全相同)—— 這個單一變因 ablation 本身是乾淨的。

### 4.3 涵蓋的方法

**A 組:normalization「類型」**(固定原始 Transformer 的 Post placement);文獻出處見 §11。

| 方法 | 統計維度(輸入 B×L×D) | 核心公式 | 出處脈絡 |
|---|---|---|---|
| **BN** BatchNorm | 沿 (B, L),逐 channel | `(x-μ_c)/σ_c · γ + β` | CNN 標配;依賴 batch 統計 |
| **LN** LayerNorm | 沿 D,逐 token | `(x-μ)/σ · γ + β` | 原始 Transformer(Post-LN) |
| **IN** InstanceNorm | 沿 L,逐樣本逐 channel | 同 BN 但每樣本獨立 | Style transfer |
| **GN** GroupNorm | 沿 (D/G, L),逐 group | channel 分 G 組做 normalization | BN 的小 batch 替代品 |
| **RMSNorm** | 沿 D,逐 token | `x/RMS(x) · γ`(不減均值) | LLaMA / T5 |
| **AdaIN** | IN + 條件**向量** modulation | `IN(x)·(1+γ(c)) + β(c)` | Style transfer;γβ 由條件生成 |
| **AdaIN-local** | IN + 條件 per-position modulation | 同 AdaIN 但**不做 average pooling** | 無論文出處,本專案設計的 granularity ablation 變體 |
| **SPADE** | IN + 條件**圖** per-position modulation | `IN(x)·(1+γ_l(c)) + β_l(c)` | Semantic image synthesis |
| **WN** WeightNorm | 作用在**權重**不在 activation | `w = g·v/‖v‖`,activation 不做 normalization | 唯一不碰 activation 者 |

**B 組:LN 的「placement 策略」**(見 §2.3 表)—— Post-LN、Pre-LN、Sandwich-LN、DeepNorm。
**C 組:`RMSNorm (Pre)`** —— 檢驗 §2.2 的推導,只有在兩者都會收斂的 Pre placement 下才驗得動(§5.3)。
**H3 對照組:`AdaIN-local (bypass 切斷)` 與 `SPADE (bypass 切斷)`** —— 把 `cond` detach 掉:
正向 re-injection 逐位元保留,反向 gradient bypass 完全切斷。

以上 15 個進主表。另有 **10 個診斷對照**不進主表(它們不是要比較優劣的 normalization):
`post_ln_st` / `post_ln_gr`(§5.5 的解耦)、`post_reinject` / `post_reinject_detach` /
`post_reinject_mlp`(§5.4 的純重注入)、`post_reinject_tok` / `post_reinject_pos` / `post_reinject_rand`
(§5.4 的 E2:`cond` 只餵 token 內容 / 只餵位置 / 凍結的隨機 per-position 向量)、
`pre_ln_gr` / `deepnorm_gr`(§5.5 的健康對照)。

### 4.4 三種量測工具

1. **逐層 gradient flow**:對殘差主幹每層之後的 hidden state $h_l$ 呼叫 `retain_grad()`,backward
   後記錄 $\mathrm{RMS}(\partial L/\partial h_l)$ —— 初始化時、訓練全程每 10 步、訓練結束時。
   主表報的 block ratio 是 `rms[1]/rms[-1]`(**排除 embedding 那一列**,理由見 §7 第 5 點)。
2. **Gradient decomposition**(檢驗 H3):`cond = x.clone()` 是獨立的 autograd 節點(數值恆等、
   不改變任何計算),`cond.grad` 即「僅經條件路徑」回到 embedding 的梯度;因為 clone 的反向是恆等,
   主幹那一份可精確算出 $g_{\text{trunk}} = g_{\text{total}} - g_{\text{cond}}$。兩條曲線分開報,
   每次 probe 記錄恆等式的數值殘差(實測最大 2.9e-08)。註:`share = 0.983` 不等於「幾乎全部繞過主幹」
   —— 近正交下這對應主幹仍佔約 18%,旁路:主幹 ≈ 5.4:1。
3. **Attention probe**(檢驗 H4):顯式計算各層 attention 矩陣,量測 attend 到「目標所在相對位置」
   (copy/char_lm:t−1;modk:t−16)的平均權重(init vs final)。基準用**同樣取 16 層最大值的實測
   init 值**(copy 0.064),解析的單層 uniform baseline(copy 0.059、modk 0.028)列為參考 ——
   16 個帶雜訊的值取最大,期望值本來就高於單層期望。probe 走的不是訓練用的程式碼路徑:`MHSA`
   訓練時用 `F.scaled_dot_product_attention`,probe 時改用手算的 `scores → softmax`,數學等價、數值不同。
   **口徑在 2026-08-29 改過一次**:現在的 `attn_to_target_*` 是每顆 head 先對 batch 與位置平均,再對
   (層, head) 取最大(head-max),問的是「有沒有任何一顆 head 學會這個 offset」;舊口徑(每層先把 4 顆 head
   平均,再取 16 層最大)另存為 `*_headmean`。§5.1–§5.3 與 8 月底以前產生的資料夾是舊口徑,§5.4 的配對版是新口徑
   (例如 `post_reinject` 新口徑 0.252、舊口徑 0.231)。

**三個指標的共同限制:**

* **block ratio 是兩點統計。** `rms[1]/rms[-1]` 只用了 17 個數字裡的 2 個,中間 15 層的非單調完全
  看不到。專案有完整剖面,更好的做法是報 log 斜率或整條剖面的極差。
* **量哪一種梯度會改變結論。** activation 梯度與參數梯度在**初始化時**方向相反(Post-LN:activation
  底/頂 1.82,參數梯度 0.69);訓練後半段則同方向,所以 §5.5 的動態崩塌不因量法而改變。
* **梯度重正規化的劑量隨訓練漂移。** `target_rms` 是固定值(在 copy 初始化時校準),而 Post-LN 的
  自然梯度尺度在訓練中會掉 3 個數量級 —— 干預的相對強度在訓練後期放大約 1300 倍。這是該干預的性質,
  不是 bug,但「只改梯度大小、不改別的」這個描述要連同這一點一起讀。

## 5. Results

### 5.1 主任務 copy(diagnostic probe)

> 16 層、1500 步、lr=3e-4、無 warmup、3 種子(mean±std,ddof=1)。
> 兩條 baseline:隨機猜測 loss = ln(64) ≈ **4.159**;attention 對前一格的 uniform baseline ≈ **0.059**。
> 「attn→前一格」= 各層中最大的「attend 到前一個位置的平均 attention 權重」。

| 配置 | 初始 block ratio `rms[1]/rms[-1]` | 最終 loss | attn→前一格 | cond_path_share(最終) |
|---|---:|---:|---:|---:|
| LN / Post-LN | 1.82 | 4.1619 ± 0.0007 | 0.064(= baseline) | — |
| RMSNorm (Post) | 1.88 | 4.1618 ± 0.0007 | 0.064(= baseline) | — |
| AdaIN (Post) | 37.52 | 4.1494 ± 0.0004 | 0.072(≈ baseline) | **0.002** |
| AdaIN-local (Post) | 37.52 | **3/3 解掉**(0.0079 / 0.0095 / 0.0088)† | 0.294 ± 0.082 | **0.983** |
| GN (Post) | 2.22 | 3.79 ± 0.39 | 0.156 ± 0.086 | — |
| IN (Post) | 38.66 | 3.58 ± 1.00 | 0.127 ± 0.111 | — |
| BN (Post) | 3.21 | 3.46 ± 0.30 | 0.154 ± 0.074 | — |
| SPADE (Post) | 32.72 | 1.11 ± 1.73 | **0.289 ± 0.045** | **0.982** |
| Pre-LN | 4.71 | **0.0003 ± 0.0000** | 0.606 ± 0.009 | — |
| Sandwich-LN | 3.65 | **0.0003 ± 0.0000** | 0.544 ± 0.002 | — |
| DeepNorm | **0.99**(最平坦) | **0.0005 ± 0.0000** | 0.913 ± 0.005 | — |
| RMSNorm (Pre) | 4.58 | **0.0003 ± 0.0000** | 0.600 ± 0.016 | — |
| AdaIN-local(bypass 切斷) | 37.52 | **2/3 解掉**(4.1588 / 0.0366 / 0.0067)† | 0.252 ± 0.171 | **0.000** |
| SPADE(bypass 切斷) | 32.72 | 1/3 解掉(4.1566 / 0.0369 / 4.1471)† | 0.147 ± 0.095 | **0.000** |

> **† 雙峰配置不要只看這張表。** AdaIN-local / SPADE / BN / IN / GN 的種子分佈是雙峰或連續譜,
> 3 種子是不具代表性的切片:`adain_loc` 這 3 顆恰好全解,15 種子的真實解掉率是 **9/15**;切斷版
> 在這 3 顆是 2/3,15 種子下同樣是 **9/15**。對明顯非單峰的三點分佈報 mean±std,平均值是任何一次
> run 都沒出現過的數,所以這幾列改報「解掉數 + 逐種子值」。**H3 的結論以 15 種子為準(§5.4)。**

> **初始 block ratio 沒有預測力 —— 而且「它不預測」本身就是一個結果。** 最平坦的 DeepNorm(0.99)
> 解掉、最不平坦的 Pre-LN(4.71)也解掉、Post-LN(1.82)夾在中間卻卡死;AdaIN 與 AdaIN-local 的
> 初始 ratio **完全相同**(37.52),結果卻是 4.1494 對 0.0088。更乾淨的反例見 §5.4:純重注入是
> zero-init 的,配對版裡同一顆種子下,它與 Post-LN 的 260 個共有張量和 step-0 logits **逐位元相同**,
> 結果卻是 15/15 對 0/15。**兩個在初始化時無法區分的模型,trainability 可以完全相反** —— 任何純粹由
> 初始化狀態算出的指標,原則上都不可能預測這個差異。

> **WN(無 activation normalization)不列入主表**:最終 loss 4.077 ± 0.011(attention 停在 baseline),
> 但失敗同時混雜**前向尺度爆炸**(初始 loss ≈ 120,logits 溢出)與**反向梯度放大**。兩個效應無法分離,
> 與主表其他配置不是同一種「病」,詳見 §7 第 3 點。

**種子層級的細節**(對解讀很重要):

- **B 組三種 path-repair 策略在 copy 上跨種子幾乎零變異**(±0.0000),每個種子都在 ~150 步內收斂。
  但這只在 copy 上成立 —— modk 上 B 組是全專案變異最大的一群(§5.2)。
- **A 組的「部分逃逸」對種子高度敏感,3 種子會誤導。** 10 種子的 stability 實驗(`results/stability/`):

  | 配置 | 解掉(<0.1) | 部分進展 | 卡死 | 逐種子最終 loss(排序) |
  |---|---:|---:|---:|---|
  | BN | **1/10** | 0/10 | 9/10 | [**0.002**, 3.155, 3.174, 3.431, 3.471, 3.500, 3.501, 3.764, 3.831, 4.028] |
  | GN | 0/10 | 1/10 | 9/10 | [2.789, 3.375, 3.825, 4.072, 4.086, 4.148, 4.153, 4.158, 4.160, 4.161] |
  | IN | 0/10 | 2/10 | 8/10 | [2.425, 2.716, 3.032, 4.160, 4.160, 4.161, 4.161, 4.161, 4.161, 4.161] |
  | SPADE | **2/10** | 4/10 | 4/10 | [0.006, 0.020, 0.125, 0.213, 0.548, 2.204, 3.107, 4.149, 4.158, 4.159] |

  兩件 3 種子看不到的事:**BN 偶爾能完全解掉 copy**(一個種子到 0.002);**SPADE 不是乾淨的雙峰**,
  10 種子顯示它鋪滿整個區間(0.006 → 4.159)。3 種子也系統性低估離散程度(BN 的 std 0.296 → 1.152)。
  主表仍報 3 種子以維持可比性,穩健性數據作為附錄。
- **LN 與 RMSNorm 是最穩定的「失敗」,但這個失敗是 learning rate 相依的,不是絕對的。**

  | 配置 | lr = 1e-4 | lr = 3e-4(主表) | lr = 1e-3 |
  |---|---:|---:|---:|
  | **LN / Post-LN** | **1.1956 ± 0.3769** | 4.1619 ± 0.0007 | 4.1633 ± 0.0002 |
  | Pre-LN | 0.0011 ± 0.0000 | 0.0003 ± 0.0000 | 0.0001 ± 0.0000 |
  | AdaIN-local | 2.7934 ± 2.3299 | 0.0088 ± 0.0008 | 0.0011 ± 0.0002 |

  lr = 1e-4 時 Post-LN 逐種子是 [1.622, 0.906, 1.059]、attention 達 [0.365, 0.455, 0.409] ——
  它**確實學會了使用 attention**,只是收斂品質遠差於 Pre-LN,所以「Post-LN 完全卡死」只在
  lr ≥ 3e-4 成立。同一張表也顯示 **AdaIN-local 的救援效果在 lr = 1e-4 上不存在甚至反向**(2.793 vs 1.196);
  15 種子下 `adain_loc` 在該 lr **0/15** 逃脫、中位數 4.152。**這是 AdaIN-local 的邊界,不是純重注入的**:
  純重注入在同一個 lr 仍是 15/15(§5.4 (e))。
- **Granularity ablation 的三方對照乾淨利落**:AdaIN(global pooling)3/3 卡死;AdaIN-local
  (唯一差別 = 不 pooling)3/3 收斂;SPADE(per-position + MLP)部分逃脫且變異大。
  15 種子下是 **0/15 對 9/15**,Fisher exact 雙尾 **p = 0.0007**。
- **逃脫判準的鬆緊會改變結論。** 用寬鬆判準(loss < 隨機水準 − 0.1)AdaIN 會被算成 4/15「逃脫」,
  但那四個值是 3.9968 / 3.7391 / 2.1639 / 0.8386 —— 前兩個只是雜訊。改用「解掉」判準(loss < 0.1)
  才是 0/15。**本文一律採用「解掉」判準。**

![初始化 gradient flow](results/copy/grad_flow_init.png)
![訓練曲線](results/copy/training_loss.png)
![訓練結束 gradient flow](results/copy/grad_flow_final.png)
![gradient flow 熱圖](results/copy/grad_flow_heatmaps.png)
![cond_path_share](results/copy/cond_path_share.png)

### 5.2 跨任務驗證:modk(long-range)與 char_lm(真實資料)

最終 loss(3 種子 mean±std;copy/modk 的隨機 baseline = ln 64 ≈ 4.159;char_lm 報 train 與 **held-out val**):

| 配置 | copy | modk(K=16) | char_lm (train) | char_lm (**val**) |
|---|---:|---:|---:|---:|
| LN / Post-LN | 4.1619 ± 0.0007 | 4.1620 ± 0.0009 | 3.3178 | 3.3469 |
| RMSNorm (Post) | 4.1618 ± 0.0007 | 4.1620 ± 0.0009 | 3.3227 | 3.3469 |
| BN (Post) | 3.46 ± 0.30 | 4.1604 ± 0.0023 | 2.7710 | 2.8050 * |
| GN (Post) | 3.79 ± 0.39 | 4.1590 ± 0.0000 | 1.7004 | 1.7165 * |
| IN (Post) | 3.58 ± 1.00 | 4.1592 ± 0.0001 | 0.3078 | 0.3145 * |
| AdaIN (Post) | 4.1494 ± 0.0004 | 4.1595 ± 0.0005 | 0.2830 | 0.2904 * |
| AdaIN-local (Post) | 0.0088 ± 0.0008 | 4.1592 ± 0.0000 | 0.2053 | 0.2339 * |
| SPADE (Post) | 1.11 ± 1.73 | 4.1593 ± 0.0003 | 0.2497 | 0.2683 * |
| WN(無 activation normalization) | 4.077 ± 0.011 | 4.1719 ± 0.0012 | 2.5834 | 2.5905 |
| Pre-LN | 0.0003 ± 0.0000 | 2.93 ± 1.28 | 1.6563 | 1.8346 |
| Sandwich-LN | 0.0003 ± 0.0000 | **0.71 ± 0.64** | **1.6115** | **1.7965** |
| DeepNorm | 0.0005 ± 0.0000 | 4.1618 ± 0.0008 | 1.7151 | 1.8839 |
| RMSNorm (Pre) | 0.0003 ± 0.0000 | 2.85 ± 1.19 | 1.6632 | 1.8405 |

held-out n-gram baseline(train split 估計、val split 評估):
`uniform 4.1744 / unigram 3.3473 / bigram 2.4819 / trigram 2.0697`(nats)。
(\* : 受**架構性 acausal leakage** 污染,見下方第 2 點,**不可作為方法優劣證據**。)

**跨任務的三個要點:**

1. **Placement 效應(H1)在真實資料上重現。** char_lm 中 Post-LN / RMSNorm 的 attention 從頭到尾
   停在 uniform baseline(0.061 vs 0.059)、val loss 卡在 3.35;Pre-LN / Sandwich / DeepNorm 則
   收斂至 1.80~1.88,attention 明顯成形(0.25~0.37)。同一種 LN、僅更換 placement,在真實資料上
   仍構成決定性差異。**附帶一個乾淨的結果:Post-LN 精確收斂到 unigram** —— val 3.3469 vs
   held-out unigram 3.3473,差 **0.0004 nats**(RMSNorm 的 val 同為 3.3469)。它精確學會了字元
   邊際分佈,然後一點別的都沒學到。

2. **char_lm 上 IN/BN 家族的低 loss 是架構性 acausal leakage,不是記憶化。** 判定靠 **generalization
   gap 的對比**:causal 配置有約 0.18 nats 的正常 gap(Pre-LN +0.178、Sandwich +0.185、DeepNorm
   +0.169),IN 家族**幾乎沒有 gap**(IN +0.007、AdaIN +0.007、SPADE +0.019、AdaIN-local +0.029),
   且 val loss 0.23~0.31 **遠低於 held-out bigram 2.4819 與 trigram 2.0697** —— 任何 causal
   character model 都不該做到。**記憶化會在驗證集上崩掉,leakage 不會。**
   直接測因果性(改動位置 t 之後的 token,量位置 t 的 logits 是否改變):

   | Normalization | train() | eval() | 判定 |
   |---|---:|---:|---|
   | LayerNorm / RMSNorm | 0.000000 | 0.000000 | causal |
   | **BatchNorm** | **0.727720** | 0.000000 | 只有 train 模式洩漏 |
   | InstanceNorm | 0.922410 | 0.922410 | 兩模式皆洩漏 |
   | GroupNorm | 0.335339 | 0.335339 | 兩模式皆洩漏 |
   | AdaIN-local / SPADE | 1.086955 / 0.868808 | 同左 | 兩模式皆洩漏 |

   **A 組 9 個配置中有 6 個**帶 acausal leakage channel(BN / IN / GN / AdaIN / AdaIN-local / SPADE)。
   佐證:**GN 的 val 1.7165 反而優於 Pre-LN 的 1.8346**,而 GN 帶有同樣的通道。
   **`model.eval()` 不是修復**:它只讓 BN 的 probe 變 causal(BN 仍用洩漏的統計量**訓練**),對 IN
   家族完全無效;90/10 切分同樣擋不住 —— 這條通道在每一次 forward pass 內部就在洩漏。

3. **modk 劃出條件重注入(H3)的邊界 —— 其效果具有任務相依性。** 在需要精確 16 步對位的 long-range
   任務上,A 組全數失敗(皆為隨機水準、attention 為 baseline 0.028),包括在 copy 上逃脫的
   AdaIN-local 與 SPADE。而且 AdaIN 系的條件路徑在 modk 上**從頭到尾根本沒有啟用**(全程最大佔比:
   AdaIN 0.0082 / AdaIN-local 0.0850 / SPADE 0.0198)—— 不是「送了梯度但不夠用」,而是「壓根沒送」。
   這是「bypass 佔比是結果不是原因」的直接證據。**純重注入也救不起來**:15 顆配對種子 0/15,與 Post-LN
   逐種子最多差 0.00013(`results/threats_modk/`,§5.4 (e))。為什麼 modk 救不起來還沒有答案 ——
   「per-position 的重注入只給每個位置它自己的內容,不能代替 attention 做 16 格的對位」說得通,但沒有被測過。
   僅有 identity-shortcut 類的 placement 取得進展,但**變異是全專案最大的**:

   | 配置 | per-seed 最終 loss | 報成 |
   |---|---|---|
   | Pre-LN | [3.415, 1.483, 3.904] | 2.93 ± 1.28 |
   | Sandwich-LN | [1.309, 0.037, 0.784] | 0.71 ± 0.64 |
   | RMSNorm (Pre) | [3.506, 1.476, 3.570] | 2.85 ± 1.19 |
   | DeepNorm | [4.161, 4.162, 4.162] | 全滅 |

   **DeepNorm 在 1500 步內亦未能逃離**(attention 未成形)—— 與其解釋方向一致:α 放大殘差、β 壓小
   分支,使 attention 分支初期貢獻極小;惟此僅為推測,亦可能只是需要更多步數。

![modk 訓練曲線](results/modk/training_loss.png)
![char_lm 訓練曲線](results/char_lm/training_loss.png)

### 5.3 §2.2 的 RMSNorm ≈ LN 預測

**不能拿 Post placement 的吻合當驗證** —— 兩者在三個任務上都同樣完全失敗、逐種子吻合到小數點後
3~4 位,不是因為 gradient flow 行為相似,而是**兩個模型都塌到同一個由資料決定的 degenerate
solution**(char_lm 上就是 unigram)。**一個從不成功的模型無法驗證關於它訓練動態的預測。**
要真的檢驗這個推導,必須在**兩者都會收斂**的 placement 下比較:

| | copy | modk | char_lm (val) | copy 初始 block ratio | copy attn→前一格 |
|---|---:|---:|---:|---:|---:|
| Pre-LN | 0.0003 ± 0.0000 | 2.93 ± 1.28 | 1.8346 | 4.71 | 0.606 ± 0.009 |
| RMSNorm (Pre) | 0.0003 ± 0.0000 | 2.85 ± 1.19 | 1.8405 | 4.58 | 0.600 ± 0.016 |

三個任務都幾乎一致,而且**兩者都收斂**。這才是該預測的有效檢驗,結論與 §2.2 的推導一致。

### 5.4 機制實驗一:救援來自正向重注入的 token 內容,反向旁路非必要

§2.4 指出 `cond = x.clone()` 同時開了正向與反向兩條路。這一節把它們拆開,結論是
**救援來自正向重注入的 token 內容;反向 gradient bypass 會加速,但不是必要的;乘性調變與條件容量也都不是必要的**。
這一節就是 TAAI 2026 論文的內容(英文摘要與對照表見最上方 Paper 一節)。(b) 之後的數字都來自**配對版**
`results/reinject_n15_paired/`(15 顆配對種子,`--deterministic`)與 2026-09 用同一批種子的補跑。

**(0) 配對設計(2026-08-28 修正)。** zero-init 的用意是讓掛上重注入的模型在 step 0 與 Post-LN 是
**同一個函數**,單一變因就是「有沒有這條路徑」。但舊版程式裡,`nn.Linear` 的建構子會先抽 RNG、之後才被歸零
—— 同一顆種子下,後面的主幹權重整批錯開(舊程式實測:260 個共有張量中 91 個不同)。修正後,零初始化的層用
`nn.utils.skip_init` 建立,真正需要隨機初始化的層(MLP 變體的第一層、SPADE 的共享 MLP)用
`torch.random.fork_rng` 包住,兩者都不再推進全域 RNG。現在同一顆種子下,重注入系與 `ln_post` 的
260 個共有張量、step-0 logits 逐位元相同,資料序列也相同(`python check_pairing.py`)。因為兩個 arm
是配對樣本,檢定改用 **McNemar**(Fisher exact 的獨立性前提不成立)。AdaIN 系把 LN 換成 IN,step 0
本來就不是同一個函數,修正只保證它們的主幹權重相同。修正之前的舊批次(`results/reinject_n15/`、
`results/h3_n15/` 等)**不是配對的**;下面 (a) 用的就是舊批次,留作研究紀錄。

**(a) 切斷反向 bypass:AdaIN-local 的逃脫率不變(舊批次)。** `cond` detach 後正向逐位元保留、`cond.grad` 恆為 None:

| 配置 | forward per-position | gradient bypass | 解掉(loss < 0.1,15 種子) |
|---|:--:|:--:|---:|
| AdaIN | ✗(mean pooling) | 結構上有,**實測從未啟用** | **0/15** |
| AdaIN-local | ✓ | 啟用(cond/trunk = 6.34) | **9/15** |
| AdaIN-local(bypass 切斷) | ✓ | **完全切斷** | **9/15** |

**9/15 對 9/15**,Fisher exact **p = 1.0000**,連解掉者的 loss 中位數都相同(0.0088);切斷版最快的
種子在**第 905 步**解掉。條件路徑的梯度佔比從 0 暴增到 0.98、且與 loss 跳水同步到同一個 probe 點
(AdaIN-local 第 1010 步、SPADE 第 1363 步)—— 同步性是真的,但那是伴隨現象,不是原因。這個干預是
**乾淨的單一路徑消融**(沒有動 loss、optimizer 或任何前向計算),null 可直接解讀。

**(b) 純重注入:配對版主結果。** Post-LN 完全不變(一般 LayerNorm、零乘性調變),只在每個 norm 點
把 embedding 經一個 zero-init 線性投影(d×d 加 bias,每點 16,512 個參數)加回殘差流:

$$x \leftarrow \mathrm{LN}(x + \mathrm{attn}(x)) + W_a(\mathrm{cond}), \qquad
  x \leftarrow \mathrm{LN}(x + \mathrm{ff}(x)) + W_b(\mathrm{cond})$$

copy、16 層、1500 步、lr 3e-4、15 顆配對種子(論文的表格由 `make_table2.py` 從 `metrics.json` 直接產生,
這裡是同一組數字):

| 配置 | 條件參數/norm 點 | 解掉(< 0.1) | McNemar p(對 Post-LN) | loss 中位數 | 逃脫步數中位數 | attn→t−1 |
|---|---:|---:|---:|---:|---:|---:|
| Post-LN | — | **0/15** | — | 4.1621 | — | 0.066 |
| AdaIN-local(逐位置乘性調變) | 33,024 | 9/15 | 0.0039 | 0.0771 | 1373 | 0.265 |
| **純重注入(加性,無調變)** | **16,512** | **15/15** | **6.1e-5** | **0.0025** | 561 | 0.252 |
| **純重注入 + bypass 切斷** | 16,512 | **15/15** | 6.1e-5 | 0.0051 | 773 | 0.248 |
| 純重注入(MLP,條件參數對齊 AdaIN-local) | 33,024 | **15/15** | 6.1e-5 | 0.0025 | 496 | 0.319 |

attn 是新口徑(§4.4),15 顆的平均;初始化時是 0.065–0.068。6.1e-5 = 2 × 0.5¹⁵,是 15 對種子能達到的最小 p 值。

* **0/15 對 15/15。** 同一顆種子、step 0 逐位元相同的兩個模型,一個全部停在隨機水準(ln 64 ≈ 4.159),
  一個全部解掉。
* **切斷反向 bypass 後仍 15/15,但每一顆種子都變慢。** 逐種子配對,15 顆全部比同種子的重注入慢 176–294 步
  (符號檢定 p = 6.1e-5;逃脫步數中位數 773 對 561)。把「逃脫步數」的閾值在 0.05–2.0 之間取 400 個值逐一
  重算,每個閾值都是 15 顆全慢,所以不是閾值挑出來的。**反向旁路有幫助,但不是必要的** —— 這也順帶證明
  detach 真的切斷了東西。
* **detach 切掉的是什麼要講精確。** 它切掉的是 $\partial L/\partial\,\mathrm{cond} = \sum_i W_i^\top g_i$
  ($g_i$ 是第 i 個 norm 點輸出的梯度):從每個 norm 點跳過中間所有 LN、直接流回 embedding table 的那一項。
  $W_a, W_b$ 自己的參數梯度照樣存在,32 個投影在切斷版裡照樣在學(每一個的權重範數都離開 0)。所以這裡
  檢驗的是 B2T 那**一類理由**(「多了一條與 LN 導數無關的梯度路徑,所以梯度傳得下去」),不是 B2T 的架構:
  本專案的旁路梯度只流進 embedding,B2T 的加在每一層的主幹上。
* **乘性調變與條件容量都不是必要的。** MLP 版的條件參數與 AdaIN-local 完全相同(33,024),15/15。
  AdaIN-local **不是因果對照** —— 它同時把 LN 換成了 IN,所以純重注入贏過它不能讀成「調變有害」,
  只能說「調變不必要」。
* **干預真的被用到,而且主要用在頂部。** 480 個 norm 點(15 種子 × 32)的重注入權重 Frobenius 範數
  最小 0.29、中位數 1.62(zero-init 的起點是 0)。四種能救的變體(完整、切斷、MLP、只有 token)
  頂部 8 個 norm 點的範數(跨種子中位數)是底部 8 個的 6.9–17.9 倍,逐種子看 15/15 顆都是頂部大於底部
  (`python analyze_layer_attn.py --wnorm-only`)。
* **(舊批次)為什麼一定要補 detach 這一格。** 配對修正之前,seed 0 的 `post_reinject` cond/trunk 佔比是
  **172.63**,AdaIN-local 只有 **6.34** —— 差了 27 倍,把 (a) 的結論直接套到純重注入是外推。
  (梯度分解只存了 seed 0,不代表 15 顆;配對版的 seed 0 對 AdaIN-local 恰好是沒逃脫的那顆,
  所以這組數字沒有在配對版重算。)

**(c) E2:重注入的是 token 內容,不是位置。** `cond = x.clone()` 裡的 `x` 同時是 token embedding 與
position embedding,所以「正向重注入」還分不出重注入的是內容還是乾淨的位置訊號 —— 對 copy 特別要緊,
解題要學會 −1 的 offset,位置訊號本身有可能就是關鍵。三個配置只換 `cond` 的來源,其餘與純重注入完全相同:

| `cond` 來源 | 內容 | 位置 | 可學 | 解掉 | loss 中位數 | attn→t−1 | 重注入權重範數中位數 |
|---|:--:|:--:|:--:|---:|---:|---:|---:|
| `tok + pos`(`post_reinject`) | ✓ | ✓ | ✓ | **15/15** | 0.0025 | 0.252 | 1.623 |
| `tok`(`post_reinject_tok`) | ✓ | ✗ | ✓ | **15/15** | 0.0022 | 0.278 | 1.539 |
| `pos`(`post_reinject_pos`) | ✗ | ✓ | ✓ | **0/15** | 4.1621 | 0.066 | 0.234 |
| 凍結的隨機 per-position 向量(`post_reinject_rand`) | ✗ | ✓ | ✗ | **0/15** | 4.1621 | 0.067 | 0.234 |

* **救援需要 token 內容。** 只有內容一樣 15/15;只重注入位置訊號,與什麼都不做無法區分 —— 逐種子 loss
  與 Post-LN 最多差 0.00052。
* **「只是多了參數」也被排除。** `pos` 與凍結隨機向量加的參數與重注入完全相同(528,384),卻 0/15。
  它們的投影權重也有在長(範數中位數 0.234,同一張表裡能救的兩列是 1.54 與 1.62),所以不是最佳化失敗,
  而是那條路徑始終沒變得有用。
* **但不能說「完全」來自內容。** `tok` 有 4 顆種子停在 0.01–0.045(完整版最大 0.0032)。這個差距不顯著
  (閾值 0.01 下的配對 McNemar p = 0.125),所以也不能說位置訊號有幫助 —— 正確的說法是「需要內容」。
* **和 detach 合起來剛好是 2×2**:

  | | 有旁路梯度流回 embedding | 沒有旁路梯度 |
  |---|---|---|
  | **前向有 token 內容** | `tok + pos` 15/15、`tok` 15/15 | `tok + pos` + detach **15/15** |
  | **前向沒有 token 內容** | `pos` **0/15**(梯度回到 pos table) | 凍結隨機向量 **0/15**(buffer,沒有梯度) |

  **救不救得起來只跟前向的 token 內容走,旁路梯度只影響快慢。** 唯一的缺口:`pos` 的旁路梯度流回的是
  pos table,不是 token table;「旁路梯度流回 token table、但前向沒有 token 內容」這一格沒有乾淨的做法。

**(d) 判準穩健性。** 主要對照(Post-LN 0/15 對重注入 15/15,含 detach 與 MLP)在「解掉」閾值 0.01–2.0
之間完全不變。但 AdaIN-local 對判準敏感(閾值 0.01 時 4/15、1.0 時 10/15),`tok` 在 0.01 時是 11/15
—— 穩健性只對主要對照成立。

**(e) 適用範圍(必須跟結論一起讀)。** 2026-09-10 用同樣 15 顆配對種子補跑(「部分進展」= loss 比隨機水準
低 0.1 以上):

* **換 learning rate:救援還在。** lr = 1e-4(copy)時重注入 **15/15**(loss 0.008–0.011),Post-LN
  **0/15**(McNemar p = 6.1e-5)。Post-LN 在這個 lr 不再停在隨機水準:15 顆全部有部分進展(0.82–3.70,
  中位數 1.06),attention 也成形(0.346),但沒有一顆解掉(`results/threats_lr1e-4/`)。
* **換任務:救援不在。** modk 上重注入 **0/15**,與 Post-LN 逐種子最多差 0.00013,連部分進展都沒有;
  正對照 Pre-LN 也沒有解掉,但 **12/15 有部分進展**(McNemar p = 4.9e-4)—— 所以不是「這個任務在預算內
  沒人解得開」,而是重注入在這裡確實沒幫上忙(`results/threats_modk/`)。為什麼,還沒有答案。
* **AdaIN-local 的邊界不同。** lr = 1e-4 下 AdaIN-local **0/15**、中位數 4.152,切斷版同樣 0/15
  (舊批次,Fisher p = 1.0000),所以「切斷 bypass 會不會變差」對 AdaIN-local 在那個區間無法檢驗;
  modk 上 AdaIN-local 與 SPADE 全部卡死(§5.2 第 3 點)。
* **旁路梯度只流進 embedding**(見 (b))。B2T 那種加在主幹 Jacobian 裡的旁路是否也不必要,**沒有測**。
* **單一任務家族、單一深度(16 層)、單一台機器。**

**(f) 兩條解題路線的 attention 不一樣。** 同一批次、同一個統計量(新口徑)、15 顆種子:Pre-LN 的 t−1
attention 是 **0.786**、DeepNorm **0.938**,重注入只有 0.252(只有 token 時 0.278)—— 同樣把 copy 解掉,
path-repair 型大約是條件重注入型的三倍。9/16 重跑 7 個配置、逐種子存下逐層值(`results/layer_attn_n15/`,
final loss 與 attention 跟配對版逐種子完全相同):

| 配置 | t−1 attention 最強的層(15 顆種子的範圍) |
|---|---|
| 純重注入 | 12–16 |
| 純重注入 + bypass 切斷 | 14–16 |
| 純重注入(MLP) | 16 |
| 只有 token | 14–16 |
| Pre-LN | 1 |
| DeepNorm | 6–9 |

重注入把 t−1 attention 放在頂部,每一顆種子都是;Pre-LN 放在第 1 層,DeepNorm 在中間。但**不是「只在頂部」**:
逐種子看,純重注入的 attention 超過 0.15 的最低層落在第 5–14 層。「重注入讓頂部幾層直接拿注入的 token 內容
解 copy,下面的主幹沒有被修好」這個解讀說得通,與 E2、權重集中在頂部、modk 救不起來都一致,但沒有直接測過。

> **結論:救援來自正向重注入的 token 內容。** 反向 gradient bypass 會加速,但不是必要的;乘性調變、
> 條件容量、多出來的參數都不是原因。AdaIN 卡死的解釋也跟著換:不是「bypass 在結構上存在但 optimizer
> 不用它」,而是 **mean pooling 把每個位置自己的 token 內容平均掉了**(舊批次實測 AdaIN 的 cond/trunk
> 僅 0.0025,旁路確實幾乎沒啟用,但那是結果不是原因)。

### 5.5 機制實驗二:拉平逐層梯度「大小」不足以救活 Post-LN

先分清楚兩個命題。**靜態版(初始化時的失衡)—— 主表就已否證,不需要任何干預**:最平坦的 DeepNorm
(0.99)解掉、最不平坦的 Pre-LN(4.71)也解掉、Post-LN(1.82)夾在中間卻卡死;AdaIN 與 AdaIN-local
數值完全相同(37.52)卻一個卡死一個解掉。換成參數梯度亦然(DeepNorm 1.00 解掉、Pre-LN 2.50 解掉、
Post-LN 0.69 卡死)。**這個指標與 trainability 沒有單調關係。**

**動態版(訓練中的崩塌)—— 這才需要干預實驗。** 崩塌本身是真的,量得精確(copy, Adam, seed 0),
而且**先於**學習失敗發生 —— 在兩者的 loss 都還在隨機水準時,底層梯度已經差了 61 倍:

| | 全程 block ratio 中位數 | 低於 0.01 的 probe 比例 | 訓練結束時 |
|---|---:|---:|---:|
| Post-LN | **3.35e-04** | **99%** | 1.46e-03 |
| Post-LN + 梯度重正規化 | 1.58 | **0%** | 1.20 |

| step | Post-LN loss | Pre-LN loss | Post 底層梯度 | Pre 底層梯度 | 倍數 |
|---:|---:|---:|---:|---:|---:|
| 10 | 4.1901 | 4.2050 | 1.04e-06 | 8.60e-06 | **8×** |
| 50 | 4.1654 | 4.1477 | 7.56e-08 | 4.64e-06 | **61×** |

所以「底層梯度小只是卡住的症狀」這個質疑不成立。**但直接干預的結果是否定的。**
`GradRenormLayerNorm`:前向 = `LayerNorm(x)`(位元完全相同),反向把每層梯度的 RMS 拉回定值 ——
**只改大小,不改方向**。跑之前先過四道閘門:前向不變(max|Δ| = 0.00e+00)、方向不變
(cos = 0.99999994)、上游梯度放大 1000× 而本層輸出不變、反向確實變平(block ratio 1.86 → 1.29,
訓練全程維持)、SGD lr 校準。

| optimizer | Post-LN | **+ 梯度重正規化** | Pre-LN | seeds |
|---|---:|---:|---:|---|
| Adam 3e-4 | 4.1618 ± 0.0007 | **4.1621 ± 0.0012** | 0.0003 | 10 |
| SGD 0.01 | 4.1618 ± 0.0007 | **4.1623 ± 0.0007** | 0.0018 | 10 |
| SGD 0.03 | 4.1642 | **4.1645** | 0.0006 | 3 |

20 顆 Post 種子全部落在隨機水準 ±0.05 內、`pre_ln` 20/20 收斂,attention 也始終停在 baseline
(0.060~0.062)。**崩塌可以被完全阻止,而阻止它之後訓練結果毫無改變。SGD 臂是必要的** ——
Adam 的更新 $m/(\sqrt{v}+\epsilon)$ 會吸收逐層梯度大小的差異,只在 Adam 下測拿到 null 證明不了什麼;
Pre-LN 在三種設定下都收斂 ⇒ SGD 臂本身是有效的。這個 null 有兩道必要的防守,兩道都做了:

* **有 headroom 的區間也沒效(排除 floor effect)。** lr = 3e-4 時 Post-LN 距隨機水準只有 +0.0030,
  貼在地板上;lr = 1e-4 時它有真正的 headroom。15 種子下:

  | lr = 1e-4(Adam,15 種子) | 中位數 | 平均 | 標準差 |
  |---|---:|---:|---:|
  | Post-LN | **1.0590** | 1.6999 | 1.0266 |
  | Post-LN + 梯度重正規化 | **1.0200** | 1.1989 | 0.3913 |

  **Mann-Whitney U 雙尾:U = 94.0,p = 0.4553 —— 不顯著。** 3 種子時看起來有效(1.987 vs 1.170)
  是基線右尾拉出來的:干預真正改變的是**變異**(標準差 1.03 → 0.39),不是位置。(該區間變異係數 60%,
  分佈 [0.8241, 3.7003];窮舉 C(15,3) = 455 個 3 種子子樣本,平均值的中央 95% 是 [0.8849, 2.7616]。)

* **干預對健康模型無害(排除「干預把任何模型都弄壞了」)。** 同一個干預施加在本來就會收斂的配置上,
  `target_rms` **逐配置自行校準**(沿用對 Post-LN 校準的 6.0e-5 等於偷偷改變整體梯度尺度):

  | 配置 | 校準值 | 原版 | + 干預 | attn→前一格 |
  |---|---:|---:|---:|---|
  | Pre-LN | 1.4881e-05 | 0.0003 | 0.0009(仍收斂) | 0.606 → 0.975 |
  | **DeepNorm** | 5.0369e-05 | 0.0005 | **0.0000** | 0.913 → **0.998** |

  DeepNorm 是最關鍵的一格:它與 Post-LN **同為 Post placement、norm 同樣坐在殘差主幹上**,唯一差別
  是 α/β 縮放,而它會收斂 —— 加上同一個干預後**變好 3.21 個數量級**。

**另一條解耦路線走不通,而走不通的原因值得記錄。** `StraightThroughLayerNorm`
(`x_st = x + (x_hat - x).detach()`,前向逐元素相同、反向換成恆等)在三個任務全部卡死
(copy 4.1627 / modk 4.1808 / char_lm val 3.3546),**但這個結論不能下,因為前提是錯的**:
straight-through 沒有把反向通路變健康,而是讓它**爆炸**(block ratio 1.82 → 3908,見 §2.3)。
那個對照組是在「前向相同 + 反向更糟」的條件下失敗的,無法回答原本的問題。

**補充量測:逐層梯度的方向一致性(gradient coherence)。** 同一個模型狀態、兩個獨立 batch 各算一次
反向,對每個 Block 的參數梯度取 cosine similarity(step ≥ 700 的中位數):

| | 底 4 層 | 中間 8 層 | 頂 4 層 | 最終 loss |
|---|---:|---:|---:|---:|
| 初始化時(三者相同) | +0.75~0.84 | +0.83~0.88 | +0.86~0.89 | — |
| Post-LN | **−0.22** | −0.10 | +0.16 | 4.1615 |
| Post-LN + 梯度重正規化 | +0.09 | +0.12 | +0.20 | 4.1627 |
| Pre-LN | +0.21 | **+0.32** | **+0.62** | 0.0003 |

三件事:**方向一致性不是初始化的性質**(step 0 時三者都在 +0.75 以上,是訓練中失去的);
**梯度重正規化沒有把它救回來**(−0.22 → +0.09,仍接近雜訊),這正好解釋了為什麼修好大小沒有用;
**大小與方向是分離的** —— Pre-LN 底 4 層的參數梯度 RMS 是 3.66e-07,比 Post-LN + 重正規化的
1.12e-03 小約 **3000 倍**,卻學得起來。(已用 float64 重算確認不是浮點雜訊:ln_post −0.1264、
pre_ln +0.3949。)**但這個量測不能建立因果**:「卡住 → 梯度不一致」與「梯度不一致 → 卡住」同樣說得通;
數字本身也不支持過強的解讀(Pre-LN 底層 +0.21 對重正規化版 +0.09 只差兩倍,真正明顯的差距在中層與
頂層)。正確的表述是:**這個觀察與「方向 / conditioning 是關鍵」一致,但沒有證明它。**

## 6. Discussion

### 6.1 逐假設檢驗

**H1(gradient path > 統計維度)— 現象成立,機制問題尚未回答,而且最被看好的答案已被排除。**
換 placement 的效應(4.16 → 0.0003,四個數量級)遠大於同一 placement 下換遍 9 種 normalization 的
效應(全部落在 1.1~4.2);DeepNorm 是最乾淨的對照 —— **不改變 Post 結構、只調整 α/β 縮放**,
就把「完全卡死」修成與 Pre-LN 相同的收斂品質。**但把「gradient path」理解為逐層梯度大小的失衡時,
該機制已被排除**(靜態版由主表否證、動態版由 §5.5 否證)。DeepNorm 與 Pre-LN 都同時改變了前向
activation 尺度、反向梯度尺度與殘差權重比例,剩下的兩個候選(前向表徵、梯度方向 / conditioning)
本專案都還沒能證明,見 §7 第 6 點。

**H2(訓練中崩塌)— 現象屬實,因果力已被否證。** 崩塌是真的且量得精確(§5.5):150 步內掉約 3 個
數量級、99% 的 probe 低於 0.01,熱圖顯示中低層在前 100 步內變暗且再未恢復,並先於學習失敗發生。
**但它不是致病機制** —— 完全阻止崩塌之後,訓練結果毫無改變。(判讀熱圖時注意:B 組在 ~200 步後也
變暗,但那伴隨 loss 收斂到 10⁻³ —— 梯度小是任務解完,不是訊號受阻。)
量法本身也要注意:Xiong et al. 的分析談的是**參數梯度**,那也是 optimizer 實際使用的量,在該量上
本專案的資料**與文獻一致**(Post-LN 0.69,往輸出端遞增;Pre-LN 遞減);換成 activation 梯度則是
1.82,**符號相反**。同一現象在兩種梯度量法下方向相反,本身值得記錄。

**H3(granularity 是關鍵)— 假設成立,機制已由直接對照確定(§5.4)。** granularity 是關鍵變因:
AdaIN(global pooling)0/15 對 AdaIN-local(唯一差別 = 不 pooling)9/15,Fisher exact **p = 0.0007**,
對照鏈完整(IN 0/10 → AdaIN 0/15 → AdaIN-local 9/15)且後兩者精確等參數。機制則是**正向重注入的
token 內容**,不是反向的 gradient bypass(配對版,§5.4):純加性、零調變的重注入 15/15 解掉(對 Post-LN
的 McNemar p = 6.1e-5);切斷反向 bypass 後仍 15/15,只是每顆種子都變慢;條件參數與 AdaIN-local 相同的
MLP 版 15/15;只重注入位置訊號則 0/15 —— **「反向 bypass」「調變容量」「多出來的參數」三個替代解釋
都被排除,而且要的是 token 內容,不是位置。** 剩餘未解耦的因素:SPADE 額外的共享 MLP 似乎讓逃脫
更不穩定,原因未探究;且 SPADE 坐在分岔點上(§7 第 7 點),不承載 H3 的結論。

**H4(attention 不可繞過)— 成立,但正確的表述是「必要而非充分」。** 所有逃脫的 run,attention 對
目標位置的權重都遠高於 uniform baseline(舊口徑,§4.4。copy:SPADE 0.24~0.32、AdaIN-local 0.24~0.39、
Pre-LN 0.61、DeepNorm 0.91;modk:有進展的 Sandwich / Pre-LN 達 0.44 / 0.39,baseline 0.028;char_lm:
收斂的 placement 達 0.25~0.37,Post-LN / RMS 停在 0.061),所有卡死的 run 都停在 baseline。
**關鍵證據是切斷 bypass 的那一組 attention 照樣成形** —— 在完全沒有 gradient bypass 的情況下解掉了題,
所以這個結論不依賴任何關於 AdaIN 的推論;這與 H3 的機制不衝突:**重注入讓模型訓練得起來,attention
仍然負責跨位置搬運。** 但 h3_n15 的 45 個 run(3 配置 × 15 種子)顯示**兩個分佈是重疊的**:解掉的
18 個 attention 落在 0.237~0.573,卡死的 27 個落在 0.059~0.340 —— AdaIN 有 run 形成了 0.34 的
attention 卻仍然卡死。所以正確的表述是**「attention 成形是解題的必要條件,不是充分條件」**,不能說
兩個分佈可以用 attention 權重分開;H4 的結論不受影響,它靠的是「解掉的 run 全部遠高於 baseline」
與「切斷 bypass 後 attention 照樣成形」。配對版再補一件事:兩種解法的 attention 不只大小不同,位置也不同
—— 重注入的 t−1 attention 最強在第 12–16 層,Pre-LN 在第 1 層,15 顆種子都是(§5.4 (f))。

### 6.2 用 gradient flow 重新分類

(此分類是本實驗觀察的整理,不是嚴格的理論分類)

| 分類 | 方法 | 行為特徵 |
|---|---|---|
| **通路修復型(path-repair)** | Pre-LN、RMSNorm (Pre)、Sandwich-LN、DeepNorm | identity shortcut 或等效縮放;**在 copy 上**跨種子零變異、快速收斂 |
| **條件重注入型(conditional re-injection)** | 純重注入(15/15)、AdaIN-local(9/15)、SPADE(部分種子) | 主幹仍堵塞,但每層直接讀到乾淨的 per-position embedding。**要的是 token 內容**(只重注入位置 0/15);**切斷反向 bypass 後逃脫率不變**(只是變慢);**乘性調變不是必要的**;copy 上 t−1 attention 集中在頂部幾層 |
| **統計補償型(statistics-compensation)** | BN、GN、IN | 改變失衡的形狀與幅度,不可靠。10 種子下 BN 有 1/10 **完全解掉**(0.002)、GN 0/10、IN 0/10 |
| **無效型(ineffective)**(**限 lr = 3e-4**) | Post-LN、RMSNorm (Post)、AdaIN(per-position 資訊被 pooling 摧毀)、WN(前向/反向尺度皆失控) | 跨種子穩定地卡在隨機水準。**但 Post-LN 在 lr = 1e-4 下有部分進展(3 種子平均 1.20)且 attention 成形,只是沒有解掉(15 種子 0/15)** |

兩個限定:**「跨種子零變異」只在 copy 上成立**(modk 上 B 組是全專案變異最大的一群);
**「無效型」是 lr 相依的**。分類表的行為特徵應理解為任務與 lr 相依的描述。

### 6.3 與文獻的關係

H1、H2 的核心結論(Post-LN 的 gradient path 病灶、Pre-LN 的 identity shortcut、DeepNorm 的修復)
**都是既有文獻的結果**:Xiong et al. (2020) 已給出理論分析,Wang et al. (2022) 已證明 DeepNorm ——
本專案做的是用一個乾淨的合成任務把這些理論**重現並視覺化**,不是新知識。自行設計的部分是四個補充
對照:**(1)** 把 BN / IN / GN / AdaIN / SPADE / WN 這些跨領域方法放進**同一個受控框架**同框對照;
**(2)** 用 **gradient decomposition** 把「條件路徑載了多少梯度」從概念變成可量測的量;
**(3)** 用 AdaIN / AdaIN-local / AdaIN-local-detach / 純重注入(配對種子)的**多方 causal ablation**,
加上 E2 的內容 / 位置分解,孤立「條件 granularity」「正向 vs 反向路徑」與「重注入的是什麼」三個變因;
**(4)** 用 **attention probe** 檢驗「繞過 attention」的替代解釋。其中 (3) 是留下來最實在的一項 —— 它
**排除了**一個看似合理、而且有強相關性支持的機制假說,並用一個更簡單的干預給出肯定式的替代答案。
這些屬於探索性的延伸量測,結論的外推性受 §7、§8 所列限制約束。

**(3) 這一項與文獻的關係(也就是 TAAI 論文的定位)。** 「每一層加回 embedding」或「加一條繞過
normalization 的連接」本身不是新東西:B2T connection(Takase et al., 2023)在層內加一條繞過 LN 的連接;
DenseFormer(Pagliardini et al., 2024)在每個 block 後,對包含 embedding 輸出 X₀ 在內的所有前層表徵
取加權平均,本專案的純重注入是這個家族裡最小的成員(只取 X₀、純加性、零初始化);looped /
recurrent-depth 模型的 input injection(例如 McLeish et al., 2024)也是每一層把 embedding 加回去。
不同的是**對「為什麼有用」的檢驗**:B2T 用反向的理由解釋它的連接(導數裡多了一個與 LN 導數無關的
單位矩陣,所以梯度傳得下去),但這種連接同時也把資訊往前送。據我們所知,還沒有研究把這兩個方向分開,
去檢驗繞過 normalization 的連接為什麼救得回訓練;本專案用配對種子加 detach 做了這件事,結論是反向
不是必要的。範圍要講清楚:本專案的旁路梯度只流進 embedding,B2T 的加在每一層的主幹上,所以測的是
B2T 那**一類理由**,不是 B2T 的架構。

其他相關工作:Xu et al. (2019) 用 detach 分開的是 **LayerNorm 本身**的前向與反向,結論是反向比較重要
—— 對象不同,結論剛好形成對照。ReZero(Bachlechner et al., 2021)用一個零初始化的**純量**去縮放既有的
殘差分支;這裡是新增一條從 embedding 出發的路徑,每個 norm 點一個 d×d 矩陣。Admin(Liu et al., 2020)
指出逐層梯度失衡不是 Post-LN 訓練困難的根本原因(他們歸因於 amplification effect),與 §5.5 的
GradRenorm null 方向一致。

### 6.4 總結

在本實驗的範圍(三個小規模任務、16 層;主表 3 種子,H3 那條線 15 種子)內,證據與以下觀點一致:
**不同 normalization 方法對 trainability 的影響差異,更多來自殘差流的結構 —— identity shortcut、
殘差縮放、per-position 條件重注入 —— 而非特徵分布的統計性質本身**。跨任務驗證進一步劃出邊界:
placement 效應在合成探針與真實資料上都重現;條件重注入的救援在 copy 上跨兩個 lr(3e-4、1e-4)成立,
換成 modk 就不成立;任務難度提高時,identity shortcut 是唯一在預算內有進展的通路設計。

**對 placement(H1)來說,「結構透過什麼路徑起作用」尚未確定,而且原本最被看好的答案已被排除。**
所以這部分應理解為對既有文獻**現象**的重現,而非對其**機制**的證明 —— 而且它額外排除了一個看似最自然的
機制解釋。條件重注入(H3)則不同:它有肯定式的答案(下面第 1 項)。

**本專案資料支持強度最高的,是以下五項具體結果**(依證據強度排序;每一項的數字都由
`verify_doc_numbers.py` 對回原始資料重算):

1. **救援來自正向重注入的 token 內容(肯定式結果)。** 配對版中,同一顆種子、step 0 逐位元相同的兩個
   模型,Post-LN **0/15**、加上零初始化的加性重注入 **15/15**(McNemar p = 6.1e-5);只重注入 token 內容
   15/15,只重注入位置訊號或凍結隨機向量 0/15(參數量完全相同);條件參數與 AdaIN-local 相同的 MLP 版
   15/15。「調變容量」與「多出來的參數」兩個替代解釋同時被排除。(§5.4)
2. **反向 gradient bypass 對逃脫「非必要」,但有幫助。** 純重注入切斷反向 bypass 後仍 **15/15**,只是
   15 顆種子全都慢了 176–294 步(p = 6.1e-5)。AdaIN-local 的旁路佔比從 0 暴增到 0.98、與 loss 跳水
   同步到同一個 probe 點,但完全切斷後逃脫率一點都沒變(9/15 對 9/15,p = 1.0,舊批次)。(§5.4)
3. **條件必須攜帶逐位置資訊。** 0/15 對 9/15,p = 0.0007,對照鏈完整且後兩者精確等參數。(§5.1)
4. **跨位置搬運由 attention 完成,但 attention 成形是必要而非充分。** 解掉的 18 個 run attention
   落在 0.237~0.573,全部遠高於 baseline 0.059;卡死的 27 個落在 0.059~0.340,**兩個分佈重疊**。
   兩種解法的 attention 也放在不同的層:重注入在頂部(12–16 層),Pre-LN 在第 1 層。(§6.1、§5.4 (f))
5. **拉平逐層梯度大小不足以救活 Post-LN。** 崩塌是真的、可以被完全阻止、而阻止它之後結果毫無改變
   (兩種 optimizer、三個 lr);有 headroom 的 lr = 1e-4 區間 15 種子下 p = 0.4553;干預對健康模型
   無害(DeepNorm 加上它反而變好 3.21 個數量級)。(§5.5)

> **兩個否定結果(第 2、5 項)的證據強度不對等,不該並列。** 第 2 項的 `cond` detach 是**單一路徑
> 消融**:沒有動 loss、optimizer 或任何前向計算,null 可以直接解讀成「那條路徑對『能不能解掉』沒有
> 因果貢獻」(對速度有,見 §5.4 (b))。
> 第 5 項的 `GradRenorm` 則**換掉了 optimizer 看到的整個更新規則**,它必須額外證明一件 detach 不需要
> 證明的事 —— **這個新規則本身不會破壞訓練**;那道對照是 §5.5 的健康對照。

**最後,一句方法論上的觀察** —— 它是討論,不是本專案的主結論:本專案兩次把一個劇烈、同步、機制上
說得通的量測當成機制(H3 的 bypass、H2 的崩塌),兩次都在直接干預之後被證明不是原因
(H3 的 bypass 只影響快慢,不決定能不能解掉)。
**強相關 + 說得通的機制 + 精確的時間同步,仍然不足以支持因果宣稱。**

## 7. 方法論註記(Threats to Validity)

1. **AdaIN/SPADE 的「條件」是自產的**(`cond` = embedding 輸出),等於人為外掛了一條路徑。這正是
   研究對象:把影像生成的 conditional normalization 移植到 autoregressive 任務時,「條件從哪來」
   本來就是設計決策;本專案用 gradient decomposition 與 detach 對照把這條路徑從**隱藏的設計副作用**
   變成**顯式的量測與干預對象**。原生應用(style image / segmentation map)中的外部條件,其路徑
   結構與此同構。

2. **BN/IN/GN 的統計沿序列計算,包含未來位置**(representation 非因果)。**copy/modk** 的 target 全在
   過去、token 獨立均勻抽樣,未來成分是與答案無關的雜訊,leakage 疑慮不成立;**char_lm** 的
   `target[t] = x[t+1]` 落在 normalization 統計的計算範圍內,存在**真實的 future leakage**,污染範圍
   是 A 組 9 個中的 6 個,且 `model.eval()` 與 train/val 切分都不是修復 —— 完整判定與因果性 probe 見
   §5.2 第 2 點。此外 representation 非因果使這些方法**不能直接用於 autoregressive 生成**,本專案的
   結論僅限 training dynamics 層面。

3. **WN 組存在混雜效應**:沒有 activation normalization 時,前向尺度失控(初始 loss ≈ 120,logits
   爆炸)與反向梯度放大同時發生,無法把失敗乾淨地歸因給 gradient flow 單一因素。正確的解讀是
   「activation normalization 同時承擔前向與反向的尺度控制」,而非單獨支持 H1。

4. **部分逃逸的中間結果(BN/GN/IN)對隨機性敏感**:主表用 3 種子報 mean±std,種子數太少(10 種子的
   穩健性數據見 §5.1),這些配置的排序不應被過度解讀。**且逃脫判準的鬆緊會改變結論** —— 用寬鬆判準
   會把只是雜訊的 3.99 也算成「逃脫」;本文一律採用「解掉」判準(loss < 0.1)。

5. **初始化時的逐層梯度比較有量綱陷阱。** 真正的問題不是「Pre/Sandwich 有 final LN 而 Post 沒有」,
   而是 embedding 那一列的量綱:把 embedding 層梯度放在分子時,它一進第一個 LN 就被除以 σ ≈ emb_std,
   反向被放大約 50 倍 —— 那樣的比值有一大半在量 `1/emb_std`。固定其他條件、只掃 embedding init std
   可以看得很清楚:

   | 配置 | emb_std 0.02 | 0.20 | 1.00 | 排除 embedding 列後 |
   |---|---:|---:|---:|---|
   | LN Post | 64.8 | 6.5 | 1.3 | 1.66 / 1.66 / 1.70 |
   | IN Post | 1592.1 | 158.0 | 34.4 | 37.15 / 37.05 / 37.15 |
   | Pre-LN | 275.0 | 26.4 | 3.4 | 4.75 / 4.69 / 2.52 |
   | DeepNorm | 35.4 | 3.5 | 0.7 | 0.99 / 0.99 / 0.99 |

   含 embedding 的比值精準隨 emb_std 反比縮放,排除那一列後幾乎不動 —— 本專案因此改用
   `rms[1]/rms[-1]`;含混淆的欄位仍保留在 `summary.csv` 的 `init_bottom_top_ratio_confounded`,
   因為「這個指標被初始化尺度混淆」本身就是一個發現。**即使換成乾淨指標,它與最終 loss 仍幾乎零相關**
   (最乾淨的反例是純重注入:配對版裡同一顆種子下,它在 step 0 與 Post-LN 逐位元相同,結果卻是 15/15 對 0/15)
   —— 這一欄的角色是描述初始狀態,不是預測工具。

6. **「前向尺度」與「反向通路」仍未完全解耦 —— 這是本設計最大的歸因缺口。** 所有成功的配置都同時
   改善了前向 activation 尺度與反向 gradient path。已排除其中一個候選(逐層梯度**大小**,§5.5),
   剩下兩個候選本專案都還沒測:**前向表徵**(Post-LN 每層都把主幹重新正規化,早期層的貢獻被反覆改寫;
   Pre-LN 的殘差流則從頭到尾保有一條未被正規化的 identity path)、**梯度的方向 / conditioning**
   (§5.5 的干預刻意保留了方向,只補做了一次純量測,結果與假設一致但**無法區分因果與症狀**)。
   要分辨這兩者,需要一個能單獨改前向、不動反向的配置。**難點是設計上的,不是算力問題。**

7. **SPADE 的結果跨環境不可攜。** SPADE 坐在分岔點上:同一台機器、同一個種子可以逐位元重現(本專案的
   `[0.2129, 3.1073, 0.0060]` 重跑時完全復現),但換 GPU / torch 版本 / kernel 選擇就可能翻到分岔的
   另一邊(在另一張 RTX 2060 上重跑得到完全不同的逐種子結果)。`requirements.txt` 已鎖版本並記錄產生
   環境,但**任何 SPADE 的逐種子數字都應視為環境專屬**。本專案的 H3 結論因此以 AdaIN-local 承重。

8. **決定性設定不等於決定性實驗。** 已啟用 `torch.use_deterministic_algorithms(warn_only=True)` 與
   `cudnn.deterministic`,並在啟動前設 `CUBLAS_WORKSPACE_CONFIG`。即便如此,
   `F.scaled_dot_product_attention` 的 memory-efficient backward **仍無決定性實作**。實測本專案的
   結果在同一環境內逐位元穩定,但這不是設定所保證的。

9. **BN 的量測協定會改變它的梯度數字,但不改變它的訓練。** 專案訓練時全程 `model.train()`,因此
   probe 時 BN 也用 batch 統計;對照見 `--probe-eval-mode`(結果在 `results/<task>_bn_evalmode/`):

   | 任務 | 訓練 loss(train 模式 → eval 模式 probe) | 初始 `rms[0]` | 倍數 |
   |---|---|---|---:|
   | copy | 3.4566 → 3.4566 | 9.11e-03 → 8.17e-01 | **89.7×** |
   | modk | 4.1604 → 4.1604 | 8.74e-03 → 9.08e-01 | **103.8×** |
   | char_lm | 2.7710 → 2.7710 | 9.53e-03 → 1.03e+00 | **107.9×** |

   **訓練 loss 三個任務都完全相同**(該旗標只作用在 probe),但 BN 的初始梯度量測差了兩個數量級。
   主表採用 train 模式的協定,但 **BN 的梯度數字應理解為協定相依,不宜與其他配置直接並列比較**。

10. **前向與反向不能完全分開。** 重注入項 $W_i\,\mathrm{cond}$ 改變了之後每一個 LN 的輸入,也就改變了
    那些 LN 的 Jacobian(1/σ 那一項)。detach 移除的只是**顯式**的旁路梯度,前向改變間接造成的反向改變
    仍然存在。所以「反向旁路不必要」指的是那條顯式、流回 embedding 的旁路梯度。

11. **舊資料夾與現在的程式不完全相容。** 兩件事都發生在 2026-08 底:**(a)** 配對修正(§5.4 (0))改變了
    多了條件模組的配置在同一顆種子下的初始權重,所以用現在的程式重跑 `copy`、`h3_n15`、`reinject_n15`
    等舊資料夾裡的 AdaIN / SPADE / 重注入配置,逐種子數字不會相同(沒有條件模組的配置不受影響);
    **(b)** attention 的口徑改成 head-max(§4.4)。舊資料夾保留原樣,作為當時結論的出處。

## 8. Limitations

* **單一規模。** 任務涵蓋 diagnostic probe(copy)、long-range dependency(modk)與小型真實資料
  (char_lm),但仍是 16 層、~3.2M 參數、單一資料集:對詞級 language modeling、翻譯、影像任務,以及
  百層 / 十億參數規模的外推性**未經驗證**。Xiong et al. (2020) 與 Wang et al. (2022) 的結果提示大
  方向一致,但本專案自身不構成證據。
* **種子數。** H3 那條線與兩個機制實驗是 15 種子,2×2 解耦是 10 種子,主表其餘配置仍是 3 種子,對
  高變異配置只夠報告現象、不足以估計逃脫機率。
* **「切斷 bypass 會不會降低逃脫率」只對純重注入有答案。** 純重注入切斷後仍 15/15,但每顆種子都變慢
  (§5.4 (b))。對 AdaIN-local(9/15 對 9/15),能下的結論是「bypass 非必要」(存在性結果,穩固),
  不包括「bypass 完全無用」—— 要分辨 60% 與 80% 的逃脫率大約需要每組 80 個種子。
* **單一深度、單一機器。** 重注入的結果只在 16 層、一台機器(RTX 5070)上跑過;8 / 32 層的深度外推
  與跨機器的可攜性都沒有測(SPADE 這類坐在分岔點上的配置跨機器會翻面,§7 第 7 點)。
* **「前向表徵」這個候選仍未測,這是最大的歸因缺口。** 逐層梯度「大小」已被排除,「梯度方向 /
  conditioning」已有一次純量測(結果與假設一致但**無法區分因果與症狀**),但「Post-LN 每層重新
  正規化主幹」這個前向候選從未被單獨測試。難點是設計上的:前向與反向綁在同一個 LN 上,拆開本身
  就是設計難題 —— 不是算力問題。
* **lr 依賴性是實質的。** Post-LN 的「完全卡死」只在 lr ≥ 3e-4 成立;lr = 1e-4 時它有部分進展
  (15 種子中位數 1.06,0/15 解掉)並形成 attention。AdaIN-local 的救援在該 lr 下不存在,純重注入的
  救援則仍是 15/15(§5.4 (e))。主表的所有結論都應理解為 lr = 3e-4 下的行為。
* **未涵蓋** warmup、gradient clipping、learning rate schedule 等實務補救手段與 normalization 的
  交互作用。
* **理論部分**只到 Jacobian 尺度分析的層級,未做 spectral radius / Lipschitz 常數的嚴格推導。
* **主表的 `git_dirty: true` 不是可追溯性缺口。** `results/copy`、`modk`、`char_lm` 等 7 個目錄的
  `provenance.json` 記著 `git_dirty: true`,那是因為舊版 `provenance.py` 在**訓練結束後**才拍快照
  —— run 把自己的輸出寫進已進版控的 `results/` 時 working tree 就髒了,而程式碼一個字都沒改。
  已查證:產生結果的 commit 與收錄結果的下一個 commit 之間沒有任何程式碼變更,且用那些 commit
  的程式碼實跑,10 個配置的 loss 軌跡與已發布結果逐位元相同。現版的快照改在訓練開始前拍。
* **配對版與之後的結果,provenance 裡沒有 git commit。** `reinject_n15_paired`、`threats_*`、
  `layer_attn_n15` 是在沒有 git 的資料夾裡跑的,`provenance.json` 記著 `git_commit: null`,程式版本
  無法從 provenance 回溯。能確認的是:用最終版程式(2026-09-16)重跑的 `layer_attn_n15`,7 個配置的
  final loss 與 attention 跟 `reinject_n15_paired` 逐種子完全相同(`verify_paper_numbers.py` 會檢查)。
* **如需要研究報告與額外驗證檔案等請詢問我**

## 9. 執行方式

```bash
pip install -r requirements.txt
```

```bash
python run_experiment.py --task copy      # 預設任務;16 層、1500 步、3 種子、自動選 CUDA
```

```bash
python run_experiment.py --task modk --modk-gap 16        # long-range dependency 任務
```

```bash
python run_experiment.py --task char_lm --val-every 100   # 真實資料;定期評估 held-out 驗證集
```

**重現 TAAI 論文**(配對版、lr = 1e-4、modk 三組重跑指令,以及不必重訓的檢查)見最上方 Paper 一節。

常用旗標:

```
--out-dir NAME        輸出到 results/NAME/,避免覆寫既有結果
--only KEY [KEY ...]  只訓練指定配置,其餘沿用既有 metrics.json(增量模式)
--seeds N             種子數(預設 3)
--lr / --optimizer    learning rate 與 optimizer(adam / sgd)
--target-rms VALUE    ln_gr 的重正規化目標值(換 placement / 換任務時必須重新校準)
--emb-std 0.2         掃描 embedding 初始化尺度(重現 §7 第 5 點的混淆現象)
--train-frac 0.9      char_lm 的 train/val 連續切分比例
--probe-eval-mode     探測時用 model.eval()(BN protocol,見 §7 第 9 點)
--steps / --layers / --device   自訂訓練步數、層數、運算裝置
--replot              不重訓,直接重畫圖表
--deterministic       啟用決定性演算法(實測不改變數值)
```

> **重跑會得到 25 個配置,主表只列 15 個。** 另外 10 個是診斷對照:`post_ln_st` / `post_ln_gr`
> (§5.5 的解耦,結果在 `results/copy_decouple/` 與 `results/decouple2_*/`)、`post_reinject` /
> `post_reinject_detach` / `post_reinject_mlp` 與 E2 的 `post_reinject_tok` / `post_reinject_pos` /
> `post_reinject_rand`(§5.4,配對版在 `results/reinject_n15_paired/`,配對修正前的舊批次在
> `results/reinject_n15/`)、`pre_ln_gr` / `deepnorm_gr`(§5.5 的健康對照,在 `results/healthy_ctrl_*/`)。
> 若要重現主表,用 `--only` 指定那 15 個 key。

> **`ln_gr` 系配置一定要搭配 `--target-rms`。** 預設值 6.0e-5 是對 **Post-LN** 在 copy 上校準的。
> 把干預搬到別的 placement 或別的任務而不重新校準,等於偷偷改變整體梯度尺度(SGD 下就是改了
> effective learning rate)。

完整重現另需在**啟動前**設定 `CUBLAS_WORKSPACE_CONFIG=:4096:8`(在 Python 裡設已太晚,CUDA context
已初始化)。驗證:

```bash
python verify_doc_numbers.py     # README 的數字
python verify_paper_numbers.py   # 論文的數字
python check_pairing.py          # 配對:各配置與 ln_post 在 step 0 逐位元相同
```

`verify_doc_numbers.py` **331 項**,把文件裡的**每一個**關鍵數字對回 `results/` 的原始資料重算 ——
它是常設清單,不是臨時腳本。`verify_paper_numbers.py` 對論文正文、表格與 caption 做同一件事(34 項)。
其他分析腳本:`make_table2.py`(從 `metrics.json` 產生論文的表格)、`analyze_layer_attn.py`
(逐層 attention 與重注入權重的深度剖面,§5.4 (f))、`paper/make_figs.py`(論文的 Fig. 1、Fig. 2)。

輸出(每個任務一個資料夾 `results/<task>/`):

| 檔案 | 內容 |
|---|---|
| `grad_flow_init.png` / `grad_flow_final.png` | 初始化時 / 訓練結束時的逐層梯度 RMS(seed 0) |
| `training_loss.png` | 訓練曲線(細線 = 逐種子,粗線 = 中位數) |
| `grad_flow_heatmaps.png` | 「層 × 訓練步」梯度熱圖(seed 0) |
| `cond_path_share.png` | 條件路徑梯度佔比隨訓練的變化(AdaIN 家族 / SPADE) |
| `summary.csv` / `metrics.json` | 數值摘要(mean±std)與完整原始數據 |
| `provenance.json` | 該組數字的出處:git commit(在 git 之外跑的是 null,見 §8)、torch/CUDA 版本、GPU、執行指令、時間 |

## 10. 檔案結構

```
Transformer-Gradient-Flow/
├── normalizations.py           normalization 的統一介面(BN/LN/IN/GN/RMS/AdaIN/AdaIN-local/
│                               SPADE/WN,外加兩個解耦用的診斷 norm)
├── model.py                    MHSA、五種 placement 的 Block、TransformerLM(含 gradient
│                               decomposition 與 attention probe 掛點,以及 cond_detach /
│                               reinject / cond_source 三個對照開關)
├── run_experiment.py           三個任務 + 訓練 + 三種量測 + 多種子 + 繪圖 + 摘要
├── analyze_grad_coherence.py   逐層梯度方向一致性量測(§5.5)
├── analyze_layer_attn.py       逐層 t−1 attention 與重注入權重的深度剖面(§5.4 (f))
├── check_pairing.py            配對檢查:各配置與 ln_post 在 step 0 逐位元相同(§5.4 (0))
├── make_table2.py              從 metrics.json 產生論文的表格
├── provenance.py               記錄 git commit / 環境 / 指令(快照在訓練開始前拍)
├── verify_doc_numbers.py       331 項 README 數字核對
├── verify_paper_numbers.py     34 項論文數字核對
├── requirements.txt
│
├── README.md                   本文件  現況總結,只寫現在成立的結論
├── paper/make_figs.py          產生論文的 Fig. 1、Fig. 2
│
├── data/                       Tiny Shakespeare 語料(不隨附,首次執行 char_lm 時自動下載)
├── results_legacy.zip          修訂前的凍結基準線(唯讀,含 MANIFEST.sha256)
└── results/
    ├── reinject_n15_paired/          論文主結果:配對版 + E2,15 顆配對種子
    ├── threats_lr1e-4/ threats_modk/ 論文 Threats:lr = 1e-4 與 modk 上的邊界,15 顆配對種子
    ├── layer_attn_n15/               逐層 attention(用最終版程式重跑 7 個配置)
    ├── copy/ modk/ char_lm/          三個任務的主表
    ├── stability/                    A 組穩健性,10 seeds
    ├── h3/ h3_n15/                   H3 三方 ablation,15 seeds
    ├── reinject_n15/                 純重注入對照,15 seeds(配對修正前的舊批次)
    ├── gr_lr1e-4_n15/                lr=1e-4 的 GradRenorm,15 seeds
    ├── healthy_ctrl_pre/ _deepnorm/  干預的健康對照
    ├── detach_lr1e-4_n15/            lr=1e-4 的 detach 對照,15 seeds
    ├── copy_lr1e-4/ copy_lr1e-3/     lr 掃描
    ├── *_decouple/ decouple2_*/      解耦對照(decouple2_*_n10 為 10 seeds)
    ├── *_bn_evalmode/                BN probe protocol 對照
    ├── grad_coherence.json           逐層梯度方向一致性
    └── calibration.json              逐配置的 target_rms 校準值
```

## 11. 參考文獻

* Ioffe & Szegedy. *Batch Normalization*. ICML 2015. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
* Ba, Kiros & Hinton. *Layer Normalization*. 2016. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
* Ulyanov, Vedaldi & Lempitsky. *Instance Normalization*. 2016. [arXiv:1607.08022](https://arxiv.org/abs/1607.08022)
* Salimans & Kingma. *Weight Normalization*. NeurIPS 2016. [arXiv:1602.07868](https://arxiv.org/abs/1602.07868)
* Huang & Belongie. *Arbitrary Style Transfer with AdaIN*. ICCV 2017. [arXiv:1703.06868](https://arxiv.org/abs/1703.06868)
* Vaswani et al. *Attention Is All You Need*(原始 Transformer,Post-LN). NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
* Wu & He. *Group Normalization*. ECCV 2018. [arXiv:1803.08494](https://arxiv.org/abs/1803.08494)
* Park, Liu, Wang & Zhu. *SPADE*. CVPR 2019. [arXiv:1903.07291](https://arxiv.org/abs/1903.07291)
* Zhang & Sennrich. *RMSNorm*. NeurIPS 2019. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
* Xiong et al. *On Layer Normalization in the Transformer Architecture*. ICML 2020. [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
* Ding et al. *CogView*(Sandwich-LN). NeurIPS 2021. [arXiv:2105.13290](https://arxiv.org/abs/2105.13290)
* Wang et al. *DeepNet: Scaling Transformers to 1,000 Layers*. 2022. [arXiv:2203.00555](https://arxiv.org/abs/2203.00555)

§6.3(TAAI 論文的定位)另外用到:

* Xu et al. *Understanding and Improving Layer Normalization*. NeurIPS 2019. [arXiv:1911.07013](https://arxiv.org/abs/1911.07013)
* Liu et al. *Understanding the Difficulty of Training Transformers*(Admin). EMNLP 2020. [arXiv:2004.08249](https://arxiv.org/abs/2004.08249)
* Bachlechner et al. *ReZero is All You Need: Fast Convergence at Large Depth*. UAI 2021. [arXiv:2003.04887](https://arxiv.org/abs/2003.04887)
* Takase et al. *B2T Connection: Serving Stability and Performance in Deep Transformers*. Findings of ACL 2023. [arXiv:2206.00330](https://arxiv.org/abs/2206.00330)
* Pagliardini et al. *DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging*. NeurIPS 2024. [arXiv:2402.02622](https://arxiv.org/abs/2402.02622)
* McLeish et al. *Transformers Can Do Arithmetic with the Right Embeddings*(input injection). NeurIPS 2024. [arXiv:2405.17399](https://arxiv.org/abs/2405.17399)

## License

MIT(見 [`LICENSE`](LICENSE))。
