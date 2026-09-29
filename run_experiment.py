import argparse
import csv
import json
import math
import os
import time
import urllib.request
from collections import namedtuple
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F

import provenance
from model import TransformerLM

plt.rcParams["axes.unicode_minus"] = False
from matplotlib.font_manager import FontProperties

def cjk(size=10):
    return FontProperties(family="Microsoft JhengHei", size=size)

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
RESULTS = BASE / "results"          # main() --task 改成 results/<task>
SHAKESPEARE_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
                   "master/data/tinyshakespeare/input.txt")

TASK_LABELS = {
    "copy": "複製前一個 token",
    "modk": "mod 加法(依賴 t-K)",
    "char_lm": "字元級語言建模(Tiny Shakespeare)",
}

Cfg = namedtuple("Cfg",
                 "key label norm placement cond_detach reinject cond_source")
# cond_detach / reinject / cond_source 的預設      既有配置不必改一行
Cfg.__new__.__defaults__ = (False, "none", "full")

#我眼睛好痛
CONFIGS = [
    # A 組
    Cfg("bn",        "BN (Post)",          "bn",    "post"),
    Cfg("ln_post",   "LN / Post-LN",       "ln",    "post"),
    Cfg("in",        "IN (Post)",          "in",    "post"),
    Cfg("gn",        "GN (Post)",          "gn",    "post"),
    Cfg("rms",       "RMSNorm (Post)",     "rms",   "post"),
    Cfg("adain",     "AdaIN (Post)",       "adain", "post"),
    Cfg("adain_loc", "AdaIN-local (Post)", "adain_local", "post"),
    Cfg("spade",     "SPADE (Post)",       "spade", "post"),
    Cfg("wn",        "WN(無激活歸一化)",  "wn",    "none"),


    # B 組
    Cfg("pre_ln",    "Pre-LN",             "ln",    "pre"),
    Cfg("sandwich",  "Sandwich-LN",        "ln",    "sandwich"),
    Cfg("deepnorm",  "DeepNorm",           "ln",    "deepnorm"),


    # C 組 RMSNorm≈LN 預測 只有在兩者都會收斂的 Pre 擺放下才得動
    # Post 擺放下兩者都完全失敗、逐種子吻合到小數 3~4 位 那是塌到同一個(?
    # 由資料決定的退化解(unigram) 不能用來驗證關於訓練動態的推導
    Cfg("pre_rms",   "RMSNorm (Pre)",      "rms",   "pre"),


    # D 組 前向尺度 vs 反向通路的解耦對照
    # 前向與 ln_post 逐位元相同 只有反向的 Jacobian 不同
    Cfg("post_ln_st", "Post-LN(前向同 LN,反向恆等)", "ln_st", "post"),

    # 前向位元不變(就是 LN) 只把逐層梯度大小拉平方向不動
    Cfg("post_ln_gr", "Post-LN(反向梯度重正規化)", "ln_gr", "post"),

    # 對照組:切斷條件路徑的反向梯度旁路 逐位元保留正向重注入
    Cfg("adain_loc_detach", "AdaIN-local (Post, bypass 切斷)","adain_local", "post", True),
    Cfg("spade_detach",     "SPADE (Post, bypass 切斷)","spade", "post", True),

    # 純重注入對照:Post-LN 完全不變(一般 LayerNorm、無條件調變) 只在每個norm 點把 embedding 經一個 zero-init 投影加回殘差流
    # 測 剩下的那個候選機制
    Cfg("post_reinject",     "Post-LN + 純重注入(線性)", "ln", "post",False, "linear"),

    # 需要排除容量不夠時
    Cfg("post_reinject_mlp", "Post-LN + 純重注入(MLP,參數對齊)", "ln", "post",False, "mlp"),

    # 排除正向條件路徑影響
    Cfg("post_reinject_detach", "Post-LN + 純重注入(bypass 切斷)", "ln", "post",
        True, "linear"),

    # E2:把 cond 的 token 內容與位置訊號拆開
    # 三者只有 cond_source 不同 其餘與 post_reinject 完全一致
    #   tok  只有內容(可學)  pos  只有位置(可學)  rand 只有位置區分度(凍結 無語意)
    # pos 正好是 rand 的可學版本 所以 tok/pos 分    內容 vs 位置 
    # pos/rand 分    可學 vs 凍結  兩軸切完 不需要第四格
    Cfg("post_reinject_tok",  "Post-LN + 重注入(只有 token 內容)", "ln", "post",
        False, "linear", "tok"),
    Cfg("post_reinject_pos",  "Post-LN + 重注入(只有位置)", "ln", "post",
        False, "linear", "pos"),
    Cfg("post_reinject_rand", "Post-LN + 重注入(固定隨機 per-position)", "ln", "post",
        False, "linear", "rand"),
    # 干預施加在健康模型上是否無害
    # 形式是 干預 X、結果沒變 => X 不是原因 其有效性完全取決於干預本身沒有引入新的失敗模式 
    # 這兩個配置必須搭配 --target-rms 逐配置校準
    Cfg("pre_ln_gr",   "Pre-LN(反向梯度重正規化)",   "ln_gr", "pre"),
    Cfg("deepnorm_gr", "DeepNorm(反向梯度重正規化)", "ln_gr", "deepnorm"),
]
CONFIG_BY_KEY = {c.key: c for c in CONFIGS}

GROUP_A = ["bn", "ln_post", "in", "gn", "rms", "adain", "adain_loc", "spade", "wn"]
GROUP_B = ["ln_post", "pre_ln", "sandwich", "deepnorm"]
GROUP_C = ["pre_ln", "pre_rms"]          # 2.2 預測的驗證格
# D 組 前向 activation 尺度 vs 反向 gradient path 的解耦
GROUP_D = ["ln_post", "post_ln_st", "post_ln_gr", "pre_ln"]
# H3 三方 causal ablation:
#   adain            正向逐位置 x(mean pooling) / 反向旁路 v
#   adain_loc        正向逐位置 v             / 反向旁路 v
#   adain_loc_detach 正向逐位置 v             / 反向旁路 x
GROUP_H3 = ["adain", "adain_loc", "adain_loc_detach", "spade", "spade_detach"]
# 純重注入:與 ln_post(沒有重注入)和 adain_loc(重注入 + 乘性調變)並排比較
GROUP_R = ["ln_post", "adain_loc", "post_reinject", "post_reinject_detach",
           "post_reinject_mlp"]
# E2:條件內容的成分分解      重注入的到底是    內容 還是    位置 
GROUP_E2 = ["ln_post", "post_reinject", "post_reinject_tok",
            "post_reinject_pos", "post_reinject_rand"]
# 健康對照:同一個干預施加在本來就會收斂的配置上
GROUP_HC = ["pre_ln", "pre_ln_gr", "deepnorm", "deepnorm_gr"]

# 有條件路徑的配置(detach 版本仍有正向條件路徑 只是反向被切斷)
# 重注入系沒有 調變(是加性的,不是乘性的) 但一樣經 cond 開了一條反向旁路
COND_KEYS = ["adain", "adain_loc", "spade", "adain_loc_detach", "spade_detach",
             "post_reinject", "post_reinject_mlp", "post_reinject_detach"]

COLORS = {c.key: plt.cm.tab20(i / len(CONFIGS)) for i, c in enumerate(CONFIGS)}

# 代辦:分檔管理




# 任務

def load_char_corpus():
    path = DATA_DIR / "tinyshakespeare.txt"
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        print(f"下載 Tiny Shakespeare 到 {path}")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    text = path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return ids, len(chars)


def ngram_baselines(ids, vocab, split, max_order=2, alpha=1.0):
    #在 train split 上估 n-gram,在 val split 上評估  held-out 才是有效判準
    #用整份語料自估自評得到的 bigram/trigram 是 訓練集熵 對 65^2 / 65^3 個
    #context 嚴重過擬合,不能當下界:本語料上 in-sample trigram 1.91,
    #held-out 卻是 2.07 一律回報 held-out 值
    
    tr = ids[:split].tolist()
    va = ids[split:].tolist()
    out = {"uniform": float(math.log(vocab))}

    back = np.full(vocab, alpha)
    for t in tr:
        back[t] += 1
    back = back / back.sum()
    out["unigram"] = float(-np.mean([math.log(back[t]) for t in va]))

    names = {1: "bigram", 2: "trigram"}
    for order in range(1, max_order + 1):
        ctx = {}
        for i in range(order, len(tr)):
            key = tuple(tr[i - order:i])
            row = ctx.get(key)
            if row is None:
                row = ctx[key] = np.full(vocab, alpha)
            row[tr[i]] += 1
        nll, cnt = 0.0, 0
        for i in range(order, len(va)):
            row = ctx.get(tuple(va[i - order:i]))
            p = (row[va[i]] / row.sum()) if row is not None else back[va[i]]
            nll -= math.log(p)
            cnt += 1
        out[names[order]] = nll / max(cnt, 1)
    return out


def build_task(args):
    #回傳 (vocab, attn_offset, batch_fn, val_batch_fn, baselines)\
    #copy / modk 每一步都從 generator 現生資料 沒有固定訓練集  不可能記憶化
    #val_batch_fn 為 None 只有 char_lm 是有限語料 要切分
    
    val_batch_fn, baselines = None, None

    if args.task == "copy":
        vocab, offset = args.vocab, 1

        def batch_fn(gen):
            x = torch.randint(0, vocab, (args.batch, args.seq_len), generator=gen)
            y = x.clone()
            y[:, 1:] = x[:, :-1]
            y[:, 0] = -100
            return x, y

    elif args.task == "modk":
        vocab, k = args.vocab, args.modk_gap
        assert k < args.seq_len
        offset = k

        def batch_fn(gen):
            x = torch.randint(0, vocab, (args.batch, args.seq_len), generator=gen)
            y = torch.full_like(x, -100)
            y[:, k:] = (x[:, k:] + x[:, :-k]) % vocab
            return x, y

    elif args.task == "char_lm":
        ids, vocab = load_char_corpus()
        offset = 1
        # 連續切分
        split = int(len(ids) * args.train_frac)
        train_hi = split - args.seq_len - 1
        val_lo, val_hi = split, len(ids) - args.seq_len - 1
        assert train_hi > 0 and val_hi > val_lo, "語料太短"

        def _sample(gen, lo, hi):
            starts = torch.randint(lo, hi, (args.batch,), generator=gen)
            x = torch.stack([ids[s:s + args.seq_len] for s in starts])
            y = torch.stack([ids[s + 1:s + 1 + args.seq_len] for s in starts])
            return x, y

        def batch_fn(gen):
            return _sample(gen, 0, train_hi)

        def val_batch_fn(gen):
            return _sample(gen, val_lo, val_hi)

        baselines = ngram_baselines(ids, vocab, split)
        baselines.update({"train_chars": split,
                          "val_chars": int(len(ids) - split),
                          "train_frac": args.train_frac})

    else:
        raise ValueError(f"未知任務: {args.task}")
    return vocab, offset, batch_fn, val_batch_fn, baselines


def attn_uniform_baseline(seq_len, offset):
    #因果遮罩下 attend 到 t-offset 的均勻注意力基線
    t = np.arange(offset, seq_len)
    return float(np.mean(1.0 / (t + 1)))


# 探測

def decompose_embedding_grad(hiddens, cond):
    #把 embedding 處的總梯度拆成    殘差主幹 與    條件旁路 兩份
    #舊版量的是 share = RMS(g_cond)/RMS(g_total),那是兩個範數相除,不是分解
    
    g_total = hiddens[0].grad
    g_cond = (cond.grad if cond.grad is not None
              else torch.zeros_like(g_total))
    g_trunk = g_total - g_cond

    def _rms(t):
        return t.pow(2).mean().sqrt().item()

    total_rms = _rms(g_total)
    resid = (g_total - g_trunk - g_cond).pow(2).mean().sqrt().item()
    return {
        "total_rms": total_rms,
        "trunk_rms": _rms(g_trunk),
        "cond_rms": _rms(g_cond),
        "cos_trunk_cond": float(F.cosine_similarity(
            g_trunk.flatten(), g_cond.flatten(), dim=0)),
        # 恆等式殘差(相對於總梯度) 紀錄每次 probe
        #     The decomposition identity was verified at every probe 
        "decomposition_error": resid / max(total_rms, 1e-12),
    }


def _decompose_or_na(model, hiddens, cond):
    #只有在 cond 真的是 x 的分支時才做梯度分解
    if getattr(model, "cond_decomposable", True):
        return decompose_embedding_grad(hiddens, cond)
    nan = float("nan")
    return {"total_rms": _rms_of(hiddens[0].grad), "trunk_rms": nan,
            "cond_rms": nan, "cos_trunk_cond": nan,
            "decomposition_error": nan}


def _rms_of(t):
    return t.pow(2).mean().sqrt().item() if t is not None else float("nan")


def probe_grad_flow(model, x, y, vocab, eval_mode=False):
    #一次 forward/backward 回傳 (loss, 逐層梯度 RMS, 梯度分解 dict)
    was_training = model.training
    if eval_mode:
        model.eval()          # 只影響 BN(改用 running stats) 不影響 autograd
    model.zero_grad(set_to_none=True)
    logits, hiddens, cond = model(x, record_hidden=True)
    loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1),
                           ignore_index=-100)
    loss.backward()
    rms = [h.grad.pow(2).mean().sqrt().item() for h in hiddens]
    dec = _decompose_or_na(model, hiddens, cond)
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return loss.item(), rms, dec


def setup_determinism():
    # 在支援的運算上啟用決定性演算法
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        print("未設定 CUBLAS_WORKSPACE_CONFIG,cuBLAS"
              "啟動前設 CUBLAS_WORKSPACE_CONFIG=:4096:8", flush=True)

#喵安 我在這留了個註解 沒有任何意義
@torch.no_grad()
def eval_val_loss(model, val_batch_fn, vocab, device, n_batches=20, seed=777):
    #固定種子的驗證集評估:所有配置在同一批 val 資料上比較
    was_training = model.training
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    tot = 0.0
    for _ in range(n_batches):
        x, y = val_batch_fn(gen)
        logits, _, _ = model(x.to(device))
        tot += F.cross_entropy(logits.reshape(-1, vocab),
                               y.to(device).reshape(-1),
                               ignore_index=-100).item()
    if was_training:
        model.train()
    return tot / n_batches


@torch.no_grad() #
def probe_attn_offset(model, x, offset, eval_mode=False):
    was_training = model.training
    if eval_mode:
        model.eval()
    weights = []
    model(x, weights_out=weights)
    head_max, head_mean = [], []
    for w in weights:  # (B, H, L, L)
        L = w.shape[-1]
        idx = torch.arange(offset, L, device=w.device)
        hit = w[..., idx, idx - offset]          # (B, H, L-offset)
        head_max.append(hit.mean(dim=(0, 2)).max().item())
        head_mean.append(hit.mean().item())
    if was_training:
        model.train()
    return head_max, head_mean


@torch.no_grad()
def reinject_weight_norms(model):
    #G4:逐 norm 點回報重注入投影的權重 Frobenius 範數
    out = []
    for blk in model.blocks:
        if getattr(blk, "reinject", "none") == "none":
            continue
        for m in blk.reinjects:
            w = m.proj.weight if hasattr(m, "proj") else m.w2.weight
            out.append(float(w.norm()))
    return out


def run_one_seed(seed, norm_name, placement, args, device, probe_batch,
                 vocab, offset, batch_fn, cond_detach=False, val_batch_fn=None,
                 reinject="none", cond_source="full"):
    torch.manual_seed(seed)
    model = TransformerLM(vocab, args.seq_len, args.d_model, args.n_heads,
                          args.d_ff, args.layers, norm_name, placement,
                          cond_detach=cond_detach,
                          emb_std=args.emb_std, reinject=reinject,
                          target_rms=getattr(args, "target_rms", None),
                          cond_source=cond_source).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    # 記於註解表117行
    if getattr(args, "optimizer", "adam") == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 註解表127行
    eval_mode = getattr(args, "probe_eval_mode", False)
    model.train()

    px, py = probe_batch
    _, init_rms, init_dec = probe_grad_flow(model, px, py, vocab,
                                            eval_mode=eval_mode)
    init_cond_rms = init_dec["cond_rms"]
    attn_prev_init, attn_prev_init_hm = probe_attn_offset(
        model, px, offset, eval_mode=eval_mode)

    gen = torch.Generator().manual_seed(1234 + seed)
    losses, probe_steps, probe_rms, probe_cond_rms = [], [], [], []
    probe_trunk_rms, probe_cos, probe_dec_err = [], [], []
    val_steps, val_losses = [], []
    val_every = getattr(args, "val_every", 0)
    diverged_step = None

    for step in range(1, args.steps + 1):
        x, y = batch_fn(gen)
        x, y = x.to(device), y.to(device)
        do_probe = (step == 1 or step % args.probe_every == 0)

        model.zero_grad(set_to_none=True)
        logits, hiddens, cond = model(x, record_hidden=do_probe)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1),
                               ignore_index=-100)
        if not math.isfinite(loss.item()):
            diverged_step = step
            break
        loss.backward()
        losses.append(loss.item())
        if do_probe:
            grad_ok = all(h.grad is not None and torch.isfinite(h.grad).all()
                          for h in hiddens)
            if grad_ok:
                dec = _decompose_or_na(model, hiddens, cond)
                probe_steps.append(step)
                probe_rms.append([h.grad.pow(2).mean().sqrt().item()
                                  for h in hiddens])
                probe_cond_rms.append(dec["cond_rms"])
                probe_trunk_rms.append(dec["trunk_rms"])
                probe_cos.append(dec["cos_trunk_cond"])
                probe_dec_err.append(dec["decomposition_error"])
            else:
                diverged_step = step
                break
        if (val_batch_fn is not None and val_every
                and (step % val_every == 0 or step == args.steps)):
            val_steps.append(step)
            val_losses.append(eval_val_loss(
                model, val_batch_fn, vocab, device,
                n_batches=args.val_batches))
        opt.step()

    final_val_loss = None
    if val_batch_fn is not None:
        final_val_loss = eval_val_loss(model, val_batch_fn, vocab, device,
                                       n_batches=args.val_batches)

    if diverged_step is None:
        _, final_rms, final_dec = probe_grad_flow(model, px, py, vocab,
                                                  eval_mode=eval_mode)
        final_cond_rms = final_dec["cond_rms"]
        final_trunk_rms = final_dec["trunk_rms"]
    else:
        final_rms = probe_rms[-1] if probe_rms else init_rms
        final_cond_rms = probe_cond_rms[-1] if probe_cond_rms else init_cond_rms
        final_trunk_rms = (probe_trunk_rms[-1] if probe_trunk_rms
                           else init_dec["trunk_rms"])
    attn_prev_final, attn_prev_final_hm = probe_attn_offset(
        model, px, offset, eval_mode=eval_mode)

    return {
        "seed": seed, "n_params": n_params, "cond_detach": cond_detach,
        "reinject": reinject, "cond_source": cond_source,
        "reinject_wnorm_final": reinject_weight_norms(model),
        "init_rms": init_rms, "final_rms": final_rms,
        "init_cond_rms": init_cond_rms, "final_cond_rms": final_cond_rms,
        "init_trunk_rms": init_dec["trunk_rms"],
        "final_trunk_rms": final_trunk_rms,
        "losses": losses, "probe_steps": probe_steps, "probe_rms": probe_rms,
        "probe_cond_rms": probe_cond_rms,
        "probe_trunk_rms": probe_trunk_rms, "probe_cos": probe_cos,
        "probe_decomposition_error": probe_dec_err,
        "max_decomposition_error": max(probe_dec_err) if probe_dec_err else 0.0,
        "attn_prev_init": attn_prev_init, "attn_prev_final": attn_prev_final,
        "attn_prev_init_headmean": attn_prev_init_hm,
        "attn_prev_final_headmean": attn_prev_final_hm,
        "val_steps": val_steps, "val_losses": val_losses,
        "final_val_loss": final_val_loss,
        "diverged_step": diverged_step,
    }


def tail_mean(ls, n=10):
    return float(np.mean(ls[-n:])) if ls else float("nan")


def run_config(cfg, args, device, probe_batch, vocab, offset, batch_fn,
               val_batch_fn=None):
    per_seed = [run_one_seed(s, cfg.norm, cfg.placement, args, device,
                             probe_batch, vocab, offset, batch_fn,
                             cond_detach=cfg.cond_detach,
                             val_batch_fn=val_batch_fn,
                             reinject=cfg.reinject,
                             cond_source=cfg.cond_source)
                for s in range(args.seeds)]
    s0 = per_seed[0]
    return {
        "key": cfg.key, "label": cfg.label, "norm": cfg.norm,
        "placement": cfg.placement, "cond_detach": cfg.cond_detach,
        "reinject": cfg.reinject, "cond_source": cfg.cond_source,
        "n_params": s0["n_params"], "seeds": args.seeds,
        "task": args.task, "attn_offset": offset,
        "emb_std": args.emb_std,
        "probe_eval_mode": getattr(args, "probe_eval_mode", False),
        "attn_baseline": attn_uniform_baseline(args.seq_len, offset),
        # seed 0 的完整軌跡(供逐層曲線 / 熱圖 / 條件佔比圖使用)
        "init_rms": s0["init_rms"], "final_rms": s0["final_rms"],
        "init_cond_rms": s0["init_cond_rms"],
        "final_cond_rms": s0["final_cond_rms"],
        "losses": s0["losses"], "probe_steps": s0["probe_steps"],
        "probe_rms": s0["probe_rms"], "probe_cond_rms": s0["probe_cond_rms"],
        "probe_trunk_rms": s0["probe_trunk_rms"], "probe_cos": s0["probe_cos"],
        "probe_decomposition_error": s0["probe_decomposition_error"],
        "init_trunk_rms": s0["init_trunk_rms"],
        "final_trunk_rms": s0["final_trunk_rms"],
        "attn_prev_init": s0["attn_prev_init"],
        "attn_prev_final": s0["attn_prev_final"],
        "attn_prev_init_headmean": s0["attn_prev_init_headmean"],
        "attn_prev_final_headmean": s0["attn_prev_final_headmean"],
        "diverged_step": s0["diverged_step"],
        # 跨種子彙總
        "losses_all": [sd["losses"] for sd in per_seed],
        "final_loss_seeds": [tail_mean(sd["losses"]) for sd in per_seed],
        # 舊指標:含 embedding 那一列 它有一大半在量
        # 1/emb_std(emb_std 放大 10 倍 數字就小 10 倍)與最終 loss 幾乎零相關
        # 保留它是因為 這個指標被初始化尺度混淆 本身也是個結果
        "init_ratio_seeds": [sd["init_rms"][0] / max(sd["init_rms"][-1], 1e-12)
                             for sd in per_seed],
        # 乾淨指標:排除 embedding 列 量的是 16 個 Block 之間真正的梯度衰減對 emb_std 幾乎不敏感
        "init_ratio_block_seeds": [sd["init_rms"][1] / max(sd["init_rms"][-1], 1e-12)
                                   for sd in per_seed],
        "attn_prev_final_seeds": [max(sd["attn_prev_final"]) for sd in per_seed],

        "attn_prev_final_layers_seeds": [sd["attn_prev_final"] for sd in per_seed],
        "attn_prev_init_layers_seeds": [sd["attn_prev_init"] for sd in per_seed],
        "attn_prev_final_headmean_layers_seeds":
            [sd["attn_prev_final_headmean"] for sd in per_seed],
        # 舊口徑(層內對 head 平均)保留下來 讓  稀釋了多少 本身可被檢查
        "attn_prev_final_headmean_seeds":
            [max(sd["attn_prev_final_headmean"]) for sd in per_seed],
        # 驗證集(僅 char_lm)訓練 loss 低有多種解釋  architecture-level
        # acausal leakage / 局部統計利用 / 記憶化續切分能大幅削弱記憶化這一項但擋不住 acausal leakage要分開報
        "val_loss_seeds": [sd["final_val_loss"] for sd in per_seed],
        "val_steps": s0["val_steps"], "val_losses": s0["val_losses"],
        # 實測基線:同樣取 16 層的最大值,與 final 的取法一致
        # 解析基線 attn_baseline 是 單層 值 拿它跟 16 層取最大比 有選擇偏差
        "attn_prev_init_seeds": [max(sd["attn_prev_init"]) for sd in per_seed],
        "attn_prev_init_headmean_seeds":
            [max(sd["attn_prev_init_headmean"]) for sd in per_seed],
        "max_decomposition_error": max(sd["max_decomposition_error"]
                                       for sd in per_seed),
        "diverged_steps": [sd["diverged_step"] for sd in per_seed],
        # G4:逐種子的重注入權重範數(訓練後) 非 reinject 配置為空 list
        "reinject_wnorm_seeds": [sd["reinject_wnorm_final"] for sd in per_seed],
    }


# 繪圖

def _plot_flow(ax, results, keys, which, title):
    drawn = []
    for k in keys:
        if k not in results:
            continue
        r = results[k]
        rms = np.asarray(r[which], dtype=float)
        depth = np.arange(len(rms))
        label = r["label"]
        style = dict(marker="o", ms=3.5, lw=1.6, ls="-")
        for pk, prev in drawn:
            if prev.shape == rms.shape:
                rel = np.median(np.abs(rms - prev) / np.maximum(np.abs(prev), 1e-30))
                if rel < 0.05:                     # 中位相對差 < 5% 視為重合
                    style.update(ls="--", lw=2.2, marker="x", ms=5)
                    label += f"(與 {results[pk]['label']} 重合)"
                    break
        ax.semilogy(depth, np.maximum(rms, 1e-12), color=COLORS[k], label=label,
                    **style)
        drawn.append((k, rms))
    ax.set_xlabel("深度 l(0 = embedding,l = 第 l 個 Block 之後)",
                  fontproperties=cjk(10))
    ax.set_ylabel(r"梯度強度 RMS$(\partial L/\partial h_l)$",
                  fontproperties=cjk(10))
    ax.set_title(title, fontproperties=cjk(12))
    ax.grid(True, which="both", alpha=0.3)
    if ax.get_legend_handles_labels()[0]:   # 該組沒有任何配置時不要叫 legend()
        ax.legend(prop=cjk(8))


def _panels(results):
    # 回傳這次結果實際涵蓋的分組面板
    cand = [(GROUP_A, "A 組:歸一化    類型 ", None),
            (GROUP_B, "B 組:LN 的    擺放策略 ", None),
            (GROUP_C, "C 組:Pre 擺放下 LN vs RMSNorm", ["pre_rms"]),
            (GROUP_D, "D 組:前向尺度 vs 反向通路", ["post_ln_st", "post_ln_gr"]),
            (GROUP_H3, "H3:條件路徑三方 ablation", ["adain_loc_detach"]),
            (GROUP_R, "H3(iii):純重注入(無乘性調變)",
             ["post_reinject", "post_reinject_mlp"]),
            (GROUP_E2, "E2:cond 的內容 / 位置分解",
             ["post_reinject_tok", "post_reinject_pos", "post_reinject_rand"]),
            (GROUP_HC, "健康對照:干預是否無害", ["pre_ln_gr", "deepnorm_gr"])]
    out = []
    for ks, title, required in cand:
        if not any(k in results for k in ks):
            continue
        if required and not any(k in results for k in required):
            continue
        out.append((ks, title))
    return out


def plot_grad_flow(results, which, fname, suffix, task_label):
    panels = _panels(results)
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(6.8 * len(panels), 5.2),
                             squeeze=False)
    for ax, (keys, title) in zip(axes[0], panels):
        _plot_flow(ax, results, keys, which, f"{title}{suffix}")
    fig.suptitle(f"Transformer 殘差主幹的梯度流 任務:{task_label}",
                 fontproperties=cjk(13))
    fig.tight_layout()
    fig.savefig(RESULTS / fname, dpi=150)
    plt.close(fig)


def plot_losses(results, task_label, chance=None):
    # 訓練曲線:細線 = 逐種子 粗線 = 中位數
    panels = _panels(results)
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(6.8 * len(panels), 5.2),
                             squeeze=False)
    for ax, (keys, title) in zip(axes[0], panels):
        for k in keys:
            if k not in results:
                continue
            r = results[k]
            runs = [l for l in r["losses_all"] if l]
            if not runs:
                continue
            n = min(len(l) for l in runs)
            arr = np.array([l[:n] for l in runs])
            steps = np.arange(1, n + 1)
            for row in arr:                      # 逐種子細線
                ax.semilogy(steps, np.maximum(row, 1e-12), lw=0.6,
                            color=COLORS[k], alpha=0.35)
            ax.semilogy(steps, np.maximum(np.median(arr, axis=0), 1e-12),
                        lw=1.8, color=COLORS[k], label=r["label"])
            for d in r["diverged_steps"]:
                if d is not None:
                    ax.axvline(d, color=COLORS[k], ls="--", lw=1, alpha=0.7)
        if chance is not None:
            ax.axhline(chance, color="gray", ls=":", lw=1.2, alpha=0.8)
        ax.set_xlabel("訓練步數", fontproperties=cjk(10))
        ax.set_ylabel("Cross-Entropy Loss(log)", fontproperties=cjk(10))
        ax.set_title(title + "(灰點線 = 隨機猜測)", fontproperties=cjk(12))
        ax.grid(True, which="both", alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(prop=cjk(8))
    fig.suptitle(f"訓練曲線(細線 = 逐種子,粗線 = 中位數) 任務:{task_label}",
                 fontproperties=cjk(13))
    fig.tight_layout()
    fig.savefig(RESULTS / "training_loss.png", dpi=150)
    plt.close(fig)


def plot_heatmaps(results, n_layers, task_label, drop_embedding_row=True):
    # 熱力圖:橫軸 = 訓練步數,縱軸 = 深度,顏色 = log10(RMS(梯度))
    keys = [c.key for c in CONFIGS if c.key in results]
    if not keys:
        return
    lo = 1 if drop_embedding_row else 0
    ncol = 4
    nrow = max(1, math.ceil(len(keys) / ncol))   # 配置數不再寫死 12
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.7 * nrow),
                             sharey=True, layout="constrained")
    axes = np.atleast_1d(axes).ravel()
    all_steps = sorted({s for k in keys for s in results[k]["probe_steps"]})
    if not all_steps:
        plt.close(fig)
        return
    vmin, vmax = -8, 1
    im = None
    for i, k in enumerate(keys):
        r, ax = results[k], axes[i]
        grid = np.full((n_layers + 1, len(all_steps)), np.nan)
        for j, s in enumerate(all_steps):
            if s in r["probe_steps"]:
                grid[:, j] = r["probe_rms"][r["probe_steps"].index(s)]
        grid = grid[lo:]
        with np.errstate(divide="ignore"):
            logg = np.log10(np.maximum(grid, 1e-12))
        im = ax.imshow(logg, aspect="auto", origin="lower", cmap="viridis",
                       vmin=vmin, vmax=vmax,
                       extent=[all_steps[0], all_steps[-1], lo, n_layers])
        title = r["label"]
        if r["diverged_step"] is not None:
            title += f"(第 {r['diverged_step']} 步發散)"
        ax.set_title(title, fontproperties=cjk(10))
        ax.set_xlabel("訓練步數", fontproperties=cjk(8))
        if i % ncol == 0:
            ax.set_ylabel("深度 l", fontproperties=cjk(9))
    for j in range(len(keys), len(axes)):
        axes[j].axis("off")
    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.01)
    cbar.set_label(r"$\log_{10}$ RMS$(\partial L/\partial h_l)$")
    note = ";已排除 l=0 embedding 列" if lo else ""
    fig.suptitle(f"訓練過程中的梯度流熱圖(暗 = 消失,亮 = 爆炸;seed 0{note})"
                 f" 任務:{task_label}", fontproperties=cjk(14))
    fig.savefig(RESULTS / "grad_flow_heatmaps.png", dpi=150)
    plt.close(fig)


def plot_cond_share(results, task_label):
    #條件路徑梯度佔比(相對於 embedding 總梯度)隨訓練的變化
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for k in COND_KEYS:
        if k not in results:
            continue
        r = results[k]
        steps = np.asarray(r["probe_steps"], dtype=float)
        cond = np.asarray(r["probe_cond_rms"], dtype=float)
        total = np.asarray([row[0] for row in r["probe_rms"]], dtype=float)
        share = cond / np.maximum(total, 1e-12)
        ax.plot(steps, share, lw=1.8, color=COLORS[k], label=r["label"])
    ax.set_xlabel("訓練步數", fontproperties=cjk(10))
    ax.set_ylabel("條件路徑梯度 RMS / embedding 總梯度 RMS",
                  fontproperties=cjk(10))
    ax.set_title(f"梯度分解:經條件調變路徑(繞過主幹)回傳的梯度佔比(seed 0)"
                 f" {task_label}", fontproperties=cjk(11))
    ax.grid(True, alpha=0.3)
    ax.legend(prop=cjk(9))
    fig.tight_layout()
    fig.savefig(RESULTS / "cond_path_share.png", dpi=150)
    plt.close(fig)


def _ms(vals, fmt=".4f"):
    #mean±std 字串
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)])
    if a.size == 0:
        return "nan"
    if a.size == 1:
        return f"{a[0]:{fmt}}"
    # n 小時慣例用樣本標準差(ddof=1) ddof=0 會系統性低估離散程度
    return f"{a.mean():{fmt}} ± {a.std(ddof=1):{fmt}}"


def write_summary(results):
    rows = []
    for c in CONFIGS:
        k = c.key
        if k not in results:
            continue
        r = results[k]
        n_div = sum(1 for d in r["diverged_steps"] if d is not None)
        cond_share, trunk_rms, cond_rms = "", "", ""
        if k in COND_KEYS and r["final_rms"][0] > 0:
            cond_share = f"{r['final_cond_rms'] / max(r['final_rms'][0], 1e-12):.3f}"
            # 直接報兩條路徑的絕對強度  範數比不是分解,兩者可互相抵銷
            trunk_rms = f"{r.get('final_trunk_rms', float('nan')):.3e}"
            cond_rms = f"{r['final_cond_rms']:.3e}"
        # G4:訓練後重注入權重的範數(跨種子與跨 norm 點取最小值 
        wn = [v for row in r.get("reinject_wnorm_seeds", []) or [] for v in row]
        rows.append({
            "key": k, "label": r["label"], "params": r["n_params"],
            "seeds": r["seeds"],
            "cond_detach": "yes" if r.get("cond_detach") else "",
            # E2:cond 餵的是什麼(full = tok+pos,即改動前的行為)
            "cond_source": r.get("cond_source", "full"),
            "reinject": r.get("reinject", "none") if
            r.get("reinject", "none") != "none" else "",
            "reinject_wnorm_min": f"{min(wn):.3f}" if wn else "",
            # 主指標:排除 embedding 列
            "init_block_ratio": _ms(r.get("init_ratio_block_seeds", []), ".2f"),
            # 附錄指標:含 embedding 列 已知被 emb_std 混淆
            "init_bottom_top_ratio_confounded":
                _ms(r["init_ratio_seeds"], ".1f"),
            "final_loss_mean10": _ms(r["final_loss_seeds"]),
            "val_loss": _ms([v for v in r.get("val_loss_seeds", [])
                             if v is not None]),
            "attn_to_target_final": _ms(r["attn_prev_final_seeds"], ".3f"),
            "attn_to_target_init": _ms(r.get("attn_prev_init_seeds", []), ".3f"),
            # 舊口徑:層內對 4 個 head 平均 會把單一專職 head 稀釋掉
            "attn_to_target_final_headmean":
                _ms(r.get("attn_prev_final_headmean_seeds", []), ".3f"),
            "attn_to_target_init_headmean":
                _ms(r.get("attn_prev_init_headmean_seeds", []), ".3f"),
            "attn_uniform_baseline": f"{r.get('attn_baseline', float('nan')):.3f}",
            # 這三欄取自 seed 0 的軌跡 不是跨種子彙總  表頭標明
            # 否則讀者無從分辨它們與同表其他欄位(mean±std)的口徑不同
            "cond_path_share_final_seed0": cond_share,
            "final_trunk_rms_seed0": trunk_rms,
            "final_cond_rms_seed0": cond_rms,
            "max_decomposition_error":
                f"{r.get('max_decomposition_error', float('nan')):.2e}",
            "diverged": f"{n_div}/{r['seeds']}" if n_div else "",
        })
    if not rows:
        print("WARNING: 沒有任何配置的結果,略過 summary.csv")
        return rows
    with open(RESULTS / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return rows


def make_all_outputs(results, n_layers, task, vocab=None):
    task_label = TASK_LABELS.get(task, task)
    chance = math.log(vocab) if (vocab and task in ("copy", "modk")) else None
    plot_grad_flow(results, "init_rms", "grad_flow_init.png", "(初始化時)",
                   task_label)
    plot_grad_flow(results, "final_rms", "grad_flow_final.png", "(訓練結束時)",
                   task_label)
    plot_losses(results, task_label, chance)
    plot_heatmaps(results, n_layers, task_label)
    plot_cond_share(results, task_label)
    write_summary(results)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------主函式在這
def main():
    global RESULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASK_LABELS), default="copy")
    ap.add_argument("--modk-gap", type=int, default=16,
                    help="modk 任務的依賴距離 K")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=512)
    ap.add_argument("--vocab", type=int, default=64,
                    help="copy / modk 的詞彙數(char_lm 由語料決定)")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--probe-every", type=int, default=10)
    ap.add_argument("--emb-std", type=float, default=0.02,
                    help="embedding 初始化 std。含 embedding 的初始梯度比對這個值"
                         "近似成反比,掃描它可重現該指標被混淆的現象")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--replot", action="store_true",
                    help="不重新訓練,直接從 results/<task>/metrics.json 重畫圖表")
    ap.add_argument("--only", nargs="*", default=None,
                    help="只訓練指定的配置 key,其餘結果沿用既有 metrics.json(增量模式)")
    ap.add_argument("--out-dir", default=None,
                    help="輸出目錄名稱(預設 = --task)。用不同名稱可避免覆寫既有結果")
    #Phase 3:以下旗標都會改變 training / 量測行為,預設一律維持舊行為,
    #讓新結果仍可與 results_legacy 及 Phase 2 的 H3 結果比較
    ap.add_argument("--train-frac", type=float, default=0.9,
                    help="char_lm 訓練集比例(連續切分,其餘為驗證集)")
    ap.add_argument("--val-every", type=int, default=0,
                    help="char_lm 每 N 步評估一次驗證集(0 = 只在最後評估一次)")
    ap.add_argument("--val-batches", type=int, default=20,
                    help="每次驗證集評估用幾個 batch")
    ap.add_argument("--probe-eval-mode", action="store_true",
                    help="探測時切 model.eval()(BN protocol revision)。"
                         "這是行為變更:BN 會改用 running stats 而非 batch 統計")
    ap.add_argument("--optimizer", choices=["adam", "sgd"], default="adam",
                    help="預設 adam(維持與既有結果的可比性)。sgd 用於解耦實驗:"
                         "Adam 會吸收逐層梯度大小的差異,SGD 不會")
    ap.add_argument("--target-rms", type=float, default=None,
                    help="ln_gr 的重正規化目標值(預設 None = 用在 copy 上對 "
                         "Post-LN 校準的 6.0e-5)。把干預施加在別的配置上時必須"
                         "逐配置重新校準,否則等於偷偷改變了整體梯度尺度")
    ap.add_argument("--deterministic", action="store_true",
                    help="啟用決定性演算法(實測不改變數值)。注意 SDPA 的 "
                         "memory-efficient backward 仍非決定性")
    args = ap.parse_args()

    if args.deterministic:
        setup_determinism()

    RESULTS = BASE / "results" / (args.out_dir or args.task)
    RESULTS.mkdir(parents=True, exist_ok=True)
    vocab, offset, batch_fn, val_batch_fn, baselines = build_task(args)

    if args.replot:
        with open(RESULTS / "metrics.json", encoding="utf-8") as f:
            results = json.load(f)
        n_layers = len(next(iter(results.values()))["init_rms"]) - 1
        make_all_outputs(results, n_layers, args.task, vocab)
        print(f"重畫完成:{RESULTS}")
        return

    #註解表193行
    prov_start = provenance.collect()
    if prov_start["git_dirty"]:
        files = prov_start.get("git_dirty_files") or []
        code = [f for f in files if f.endswith(".py")]
        print(f"WARNING: 開跑時 working tree 就有 {len(files)} 個未提交的改動"
              f"{'(含 ' + str(len(code)) + ' 個 .py)' if code else '(不含 .py)'}"
              f":{', '.join(files[:5])}{' …' if len(files) > 5 else ''}",
              flush=True)
        if code:
            print("先 commit 再跑。", flush=True)

    device = torch.device(args.device)
    if baselines:
        print("held-out n-gram 基線(train split 估計,val split 評估):", flush=True)
        for k in ("uniform", "unigram", "bigram", "trigram"):
            if k in baselines:
                print(f"    {k:8s} {baselines[k]:.4f} nats", flush=True)
        print(f"    train {baselines['train_chars']:,} 字元 / "
              f"val {baselines['val_chars']:,} 字元", flush=True)
    print(f"task={args.task}, vocab={vocab}, attn_offset={offset}, "
          f"attn_baseline={attn_uniform_baseline(args.seq_len, offset):.3f}",
          flush=True)
    print(f"device={device}, layers={args.layers}, steps={args.steps}, "
          f"seeds={args.seeds}", flush=True)

    # 固定探測 batch:所有配置 所有種子都在同一批資料上量測
    pgen = torch.Generator().manual_seed(999)
    px, py = batch_fn(pgen)
    px, py = px.to(device), py.to(device)

    results = {}
    if args.only and (RESULTS / "metrics.json").exists():
        with open(RESULTS / "metrics.json", encoding="utf-8") as f:
            results = json.load(f)
    for cfg in CONFIGS:
        if args.only and cfg.key not in args.only:
            continue
        if cfg.norm == "ln_gr" and args.task != "copy":
            print(f"WARNING: [{cfg.key}] 的 target_rms 是在 copy 上校準的,"
                  f"在 {args.task} 上未重新校準(實測 char_lm 的自然尺度約差 1.6 倍)。"
                  f"先重新校準。", flush=True)
        if (cfg.norm == "ln_gr" and cfg.placement != "post"
                and args.target_rms is None):
            print(f"WARNING: [{cfg.key}] 是 {cfg.placement} 擺放,但沒有給 --target-rms"
                  f"會用對 Post-LN 校準的預設值 6.0e-5。"
                  f"先重新校準。",
                  flush=True)
        t0 = time.time()
        r = run_config(cfg, args, device, (px, py), vocab, offset, batch_fn,
                       val_batch_fn=val_batch_fn)
        if baselines:
            r["ngram_baselines"] = baselines
        results[cfg.key] = r
        key = cfg.key
        # 分解恆等式必須逐 probe 成立 否則 g_trunk / g_cond 的解讀不成立
        if r["max_decomposition_error"] > 1e-6:
            print(f"WARNING: [{key}] 梯度分解恆等式殘差過大 "
                  f"{r['max_decomposition_error']:.2e} > 1e-6", flush=True)
        n_div = sum(1 for d in r["diverged_steps"] if d is not None)
        status = (f"DIVERGED {n_div}/{r['seeds']}" if n_div
                  else f"final_loss={_ms(r['final_loss_seeds'])}")
        vals = [v for v in r.get("val_loss_seeds", []) if v is not None]
        vstr = f"  val={_ms(vals)}" if vals else ""
        print(f"[{key:16s}] {time.time()-t0:6.1f}s  "
              f"attn_tgt={_ms(r['attn_prev_final_seeds'], '.3f')}  {status}{vstr}",
              flush=True)

    make_all_outputs(results, args.layers, args.task, vocab)
    with open(RESULTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    provenance.write(RESULTS, extra={"args": vars(args)}, info=prov_start)
    print(f"\n圖表與數據已輸出到 {RESULTS}")


if __name__ == "__main__":
    main()
