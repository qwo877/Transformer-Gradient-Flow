import argparse
import csv
import json
import math
import time
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F

from model import TransformerLM

plt.rcParams["axes.unicode_minus"] = False
from matplotlib.font_manager import FontProperties

def cjk(size=10):
    return FontProperties(family="Microsoft JhengHei", size=size)

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
RESULTS = BASE / "results"          # main() 會依 --task 改成 results/<task>
SHAKESPEARE_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
                   "master/data/tinyshakespeare/input.txt")

TASK_LABELS = {
    "copy": "複製前一個 token",
    "modk": "mod 加法(依賴 t-K)",
    "char_lm": "字元級語言建模(Tiny Shakespeare)",
}

# (key, 圖例標籤, norm_name, placement)
CONFIGS = [
    # A 組
    ("bn",       "BN (Post)",        "bn",    "post"),
    ("ln_post",  "LN / Post-LN",     "ln",    "post"),
    ("in",       "IN (Post)",        "in",    "post"),
    ("gn",       "GN (Post)",        "gn",    "post"),
    ("rms",      "RMSNorm (Post)",   "rms",   "post"),
    ("adain",    "AdaIN (Post)",     "adain", "post"),
    ("adain_loc", "AdaIN-local (Post)", "adain_local", "post"),
    ("spade",    "SPADE (Post)",     "spade", "post"),
    ("wn",       "WN(無激活歸一化)", "wn",    "none"),
    # B 組
    ("pre_ln",   "Pre-LN",           "ln",    "pre"),
    ("sandwich", "Sandwich-LN",      "ln",    "sandwich"),
    ("deepnorm", "DeepNorm",         "ln",    "deepnorm"),
]
GROUP_A = ["bn", "ln_post", "in", "gn", "rms", "adain", "adain_loc", "spade", "wn"]
GROUP_B = ["ln_post", "pre_ln", "sandwich", "deepnorm"]
COND_KEYS = ["adain", "adain_loc", "spade"]  # 有條件調變路徑的配置

COLORS = {k: plt.cm.tab20(i / len(CONFIGS)) for i, (k, *_ ) in enumerate(CONFIGS)}


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


def build_task(args):
    #回傳 (vocab, attn_offset, batch_fn) attn_offset = 解題所需資訊所在的相對位置(注意力探測用)
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
        n = len(ids) - args.seq_len - 1

        def batch_fn(gen):
            starts = torch.randint(0, n, (args.batch,), generator=gen)
            x = torch.stack([ids[s:s + args.seq_len] for s in starts])
            y = torch.stack([ids[s + 1:s + 1 + args.seq_len] for s in starts])
            return x, y

    else:
        raise ValueError(f"未知任務: {args.task}")
    return vocab, offset, batch_fn


def attn_uniform_baseline(seq_len, offset):
    #因果遮罩下 attend 到 t-offset 的均勻注意力基線
    t = np.arange(offset, seq_len)
    return float(np.mean(1.0 / (t + 1)))


# 探測

def probe_grad_flow(model, x, y, vocab):
    #一次 forward/backward 回傳 (loss, 逐層梯度 RMS, 條件路徑梯度 RMS)
    model.zero_grad(set_to_none=True)
    logits, hiddens, cond = model(x, record_hidden=True)
    loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1),
                           ignore_index=-100)
    loss.backward()
    rms = [h.grad.pow(2).mean().sqrt().item() for h in hiddens]
    cond_rms = (cond.grad.pow(2).mean().sqrt().item()
                if cond.grad is not None else 0.0)
    model.zero_grad(set_to_none=True)
    return loss.item(), rms, cond_rms


@torch.no_grad() #
def probe_attn_offset(model, x, offset):
    #回傳各層 attend 到 t-offset 位置的平均注意力權重
    weights = []
    model(x, weights_out=weights)
    vals = []
    for w in weights:  # (B, H, L, L)
        L = w.shape[-1]
        idx = torch.arange(offset, L, device=w.device)
        vals.append(w[..., idx, idx - offset].mean().item())
    return vals


def run_one_seed(seed, norm_name, placement, args, device, probe_batch,
                 vocab, offset, batch_fn):
    torch.manual_seed(seed)
    model = TransformerLM(vocab, args.seq_len, args.d_model, args.n_heads,
                          args.d_ff, args.layers, norm_name, placement).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    px, py = probe_batch
    _, init_rms, init_cond_rms = probe_grad_flow(model, px, py, vocab)
    attn_prev_init = probe_attn_offset(model, px, offset)

    gen = torch.Generator().manual_seed(1234 + seed)
    losses, probe_steps, probe_rms, probe_cond_rms = [], [], [], []
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
                probe_steps.append(step)
                probe_rms.append([h.grad.pow(2).mean().sqrt().item()
                                  for h in hiddens])
                probe_cond_rms.append(
                    cond.grad.pow(2).mean().sqrt().item()
                    if cond.grad is not None else 0.0)
            else:
                diverged_step = step
                break
        opt.step()

    if diverged_step is None:
        _, final_rms, final_cond_rms = probe_grad_flow(model, px, py, vocab)
    else:
        final_rms = probe_rms[-1] if probe_rms else init_rms
        final_cond_rms = probe_cond_rms[-1] if probe_cond_rms else init_cond_rms
    attn_prev_final = probe_attn_offset(model, px, offset)

    return {
        "seed": seed, "n_params": n_params,
        "init_rms": init_rms, "final_rms": final_rms,
        "init_cond_rms": init_cond_rms, "final_cond_rms": final_cond_rms,
        "losses": losses, "probe_steps": probe_steps, "probe_rms": probe_rms,
        "probe_cond_rms": probe_cond_rms,
        "attn_prev_init": attn_prev_init, "attn_prev_final": attn_prev_final,
        "diverged_step": diverged_step,
    }


def tail_mean(ls, n=10):
    return float(np.mean(ls[-n:])) if ls else float("nan")


def run_config(key, label, norm_name, placement, args, device, probe_batch,
               vocab, offset, batch_fn):
    per_seed = [run_one_seed(s, norm_name, placement, args, device, probe_batch,
                             vocab, offset, batch_fn)
                for s in range(args.seeds)]
    s0 = per_seed[0]
    return {
        "key": key, "label": label, "norm": norm_name, "placement": placement,
        "n_params": s0["n_params"], "seeds": args.seeds,
        "task": args.task, "attn_offset": offset,
        "attn_baseline": attn_uniform_baseline(args.seq_len, offset),
        # seed 0 的完整軌跡(供逐層曲線 / 熱圖 / 條件佔比圖使用)
        "init_rms": s0["init_rms"], "final_rms": s0["final_rms"],
        "init_cond_rms": s0["init_cond_rms"],
        "final_cond_rms": s0["final_cond_rms"],
        "losses": s0["losses"], "probe_steps": s0["probe_steps"],
        "probe_rms": s0["probe_rms"], "probe_cond_rms": s0["probe_cond_rms"],
        "attn_prev_init": s0["attn_prev_init"],
        "attn_prev_final": s0["attn_prev_final"],
        "diverged_step": s0["diverged_step"],
        # 跨種子彙總
        "losses_all": [sd["losses"] for sd in per_seed],
        "final_loss_seeds": [tail_mean(sd["losses"]) for sd in per_seed],
        "init_ratio_seeds": [sd["init_rms"][0] / max(sd["init_rms"][-1], 1e-12)
                             for sd in per_seed],
        "attn_prev_final_seeds": [max(sd["attn_prev_final"]) for sd in per_seed],
        "diverged_steps": [sd["diverged_step"] for sd in per_seed],
    }


# 繪圖

def _plot_flow(ax, results, keys, which, title):
    for k in keys:
        if k not in results:
            continue
        r = results[k]
        rms = np.asarray(r[which], dtype=float)
        depth = np.arange(len(rms))
        ax.semilogy(depth, np.maximum(rms, 1e-12), marker="o", ms=3.5,
                    lw=1.6, color=COLORS[k], label=r["label"])
    ax.set_xlabel("深度 l(0 = embedding,l = 第 l 個 Block 之後)",
                  fontproperties=cjk(10))
    ax.set_ylabel(r"梯度強度 RMS$(\partial L/\partial h_l)$",
                  fontproperties=cjk(10))
    ax.set_title(title, fontproperties=cjk(12))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(prop=cjk(8))


def plot_grad_flow(results, which, fname, suffix, task_label):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    _plot_flow(axes[0], results, GROUP_A, which,
               f"A 組:歸一化「類型」比較{suffix}")
    _plot_flow(axes[1], results, GROUP_B, which,
               f"B 組:LN 的「擺放策略」比較{suffix}")
    fig.suptitle(f"Transformer 殘差主幹的梯度流 —— 任務:{task_label}",
                 fontproperties=cjk(13))
    fig.tight_layout()
    fig.savefig(RESULTS / fname, dpi=150)
    plt.close(fig)


def plot_losses(results, task_label, chance=None):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for ax, keys, title in [(axes[0], GROUP_A, "A 組:歸一化類型"),
                            (axes[1], GROUP_B, "B 組:擺放策略")]:
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
            ax.semilogy(steps, arr.mean(axis=0), lw=1.4, color=COLORS[k],
                        label=r["label"])
            if arr.shape[0] > 1:
                ax.fill_between(steps, arr.min(axis=0), arr.max(axis=0),
                                color=COLORS[k], alpha=0.15, lw=0)
            for d in r["diverged_steps"]:
                if d is not None:
                    ax.axvline(d, color=COLORS[k], ls="--", lw=1, alpha=0.7)
        if chance is not None:
            ax.axhline(chance, color="gray", ls=":", lw=1.2, alpha=0.8)
        ax.set_xlabel("訓練步數", fontproperties=cjk(10))
        ax.set_ylabel("Cross-Entropy Loss(log)", fontproperties=cjk(10))
        ax.set_title(title + "(陰影 = 種子間範圍;灰點線 = 隨機猜測)",
                     fontproperties=cjk(12))
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(prop=cjk(8))
    fig.suptitle(f"訓練曲線(實線 = 種子平均)—— 任務:{task_label}",
                 fontproperties=cjk(13))
    fig.tight_layout()
    fig.savefig(RESULTS / "training_loss.png", dpi=150)
    plt.close(fig)


def plot_heatmaps(results, n_layers, task_label):
    keys = [k for k, *_ in CONFIGS if k in results]
    fig, axes = plt.subplots(3, 4, figsize=(18, 11), sharey=True,
                             layout="constrained")
    axes = axes.ravel()
    all_steps = sorted({s for k in keys for s in results[k]["probe_steps"]})
    vmin, vmax = -8, 1
    im = None
    for i, k in enumerate(keys):
        r, ax = results[k], axes[i]
        grid = np.full((n_layers + 1, len(all_steps)), np.nan)
        for j, s in enumerate(all_steps):
            if s in r["probe_steps"]:
                grid[:, j] = r["probe_rms"][r["probe_steps"].index(s)]
        with np.errstate(divide="ignore"):
            logg = np.log10(np.maximum(grid, 1e-12))
        im = ax.imshow(logg, aspect="auto", origin="lower", cmap="viridis",
                       vmin=vmin, vmax=vmax,
                       extent=[all_steps[0], all_steps[-1], 0, n_layers])
        title = r["label"]
        if r["diverged_step"] is not None:
            title += f"(第 {r['diverged_step']} 步發散)"
        ax.set_title(title, fontproperties=cjk(10))
        ax.set_xlabel("訓練步數", fontproperties=cjk(8))
        if i % 4 == 0:
            ax.set_ylabel("深度 l", fontproperties=cjk(9))
    for j in range(len(keys), len(axes)):
        axes[j].axis("off")
    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.01)
    cbar.set_label(r"$\log_{10}$ RMS$(\partial L/\partial h_l)$")
    fig.suptitle(f"訓練過程中的梯度流熱圖(暗 = 消失,亮 = 爆炸;seed 0)"
                 f"—— 任務:{task_label}", fontproperties=cjk(14))
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
                 f"—— {task_label}", fontproperties=cjk(11))
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
    return f"{a.mean():{fmt}} ± {a.std(ddof=0):{fmt}}"


def write_summary(results):
    rows = []
    for k, *_ in CONFIGS:
        if k not in results:
            continue
        r = results[k]
        n_div = sum(1 for d in r["diverged_steps"] if d is not None)
        cond_share = ""
        if k in COND_KEYS and r["final_rms"][0] > 0:
            cond_share = f"{r['final_cond_rms'] / max(r['final_rms'][0], 1e-12):.3f}"
        rows.append({
            "key": k, "label": r["label"], "params": r["n_params"],
            "seeds": r["seeds"],
            "init_bottom_top_ratio": _ms(r["init_ratio_seeds"], ".1f"),
            "final_loss_mean10": _ms(r["final_loss_seeds"]),
            "attn_to_target_final": _ms(r["attn_prev_final_seeds"], ".3f"),
            "attn_uniform_baseline": f"{r.get('attn_baseline', float('nan')):.3f}",
            "cond_path_share_final": cond_share,
            "diverged": f"{n_div}/{r['seeds']}" if n_div else "",
        })
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
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--replot", action="store_true",
                    help="不重新訓練,直接從 results/<task>/metrics.json 重畫圖表")
    ap.add_argument("--only", nargs="*", default=None,
                    help="只訓練指定的配置 key,其餘結果沿用既有 metrics.json(增量模式)")
    args = ap.parse_args()

    RESULTS = BASE / "results" / args.task
    RESULTS.mkdir(parents=True, exist_ok=True)
    vocab, offset, batch_fn = build_task(args)

    if args.replot:
        with open(RESULTS / "metrics.json", encoding="utf-8") as f:
            results = json.load(f)
        n_layers = len(next(iter(results.values()))["init_rms"]) - 1
        make_all_outputs(results, n_layers, args.task, vocab)
        print(f"重畫完成:{RESULTS}")
        return

    device = torch.device(args.device)
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
    for key, label, norm_name, placement in CONFIGS:
        if args.only and key not in args.only:
            continue
        t0 = time.time()
        r = run_config(key, label, norm_name, placement, args, device,
                       (px, py), vocab, offset, batch_fn)
        results[key] = r
        n_div = sum(1 for d in r["diverged_steps"] if d is not None)
        status = (f"DIVERGED {n_div}/{r['seeds']}" if n_div
                  else f"final_loss={_ms(r['final_loss_seeds'])}")
        print(f"[{key:9s}] {time.time()-t0:6.1f}s  "
              f"attn_tgt={_ms(r['attn_prev_final_seeds'], '.3f')}  {status}",
              flush=True)

    make_all_outputs(results, args.layers, args.task, vocab)
    with open(RESULTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\n圖表與數據已輸出到 {RESULTS}")


if __name__ == "__main__":
    main()
