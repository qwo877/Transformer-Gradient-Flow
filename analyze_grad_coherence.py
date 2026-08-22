"""逐層梯度的 方向一致性 量測(gradient coherence)
量法(純量測,吳干預):
    同一個模型狀態、兩個獨立的 batch,各算一次反向,
    對每一層的參數梯度取 cosine similarity。
      cos -> 1   該層的梯度方向在不同資料上一致 = 有訊號
      cos -> 0   方向隨 batch 亂跳 = 雜訊主導
量的是 參數梯度 而非 activation 梯度,因為參數梯度活在固定的空間裡,
跨 batch 可以直接比較,而且它才是 optimizer 真正使用的量

用法:
    python analyze_grad_coherence.py                 # 預設 3 配置 x 1500 步
    python analyze_grad_coherence.py --steps 300     # 快速檢查
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

import run_experiment as R          # noqa: E402
from model import TransformerLM     # noqa: E402


def block_param_grads(model):
    #把每個 Block 的所有參數梯度攤平成一個向量,回傳 list(長度 = 層數)
    out = []
    for blk in model.blocks:
        gs = [p.grad.flatten() for p in blk.parameters() if p.grad is not None]
        out.append(torch.cat(gs) if gs else None)
    return out


def grads_on(model, x, y, vocab):
    model.zero_grad(set_to_none=True)
    logits, _, _ = model(x)
    F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1),
                    ignore_index=-100).backward()
    g = block_param_grads(model)
    model.zero_grad(set_to_none=True)
    return g


def coherence(model, batch_a, batch_b, vocab):
    """兩個獨立 batch 的逐層參數梯度 cosine similarity。"""
    ga = grads_on(model, *batch_a, vocab)
    gb = grads_on(model, *batch_b, vocab)
    cos, mag = [], []
    for a, b in zip(ga, gb):
        if a is None or b is None:
            cos.append(float("nan")); mag.append(float("nan")); continue
        cos.append(float(F.cosine_similarity(a, b, dim=0)))
        mag.append(float(((a + b) / 2).pow(2).mean().sqrt()))
    return cos, mag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+",
                    default=["ln_post", "post_ln_gr", "pre_ln"])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/grad_coherence.json")
    cli = ap.parse_args()

    args = argparse.Namespace(
        task="copy", modk_gap=16, steps=cli.steps, seeds=1, layers=16,
        d_model=128, n_heads=4, d_ff=512, vocab=64, seq_len=64, batch=32,
        lr=3e-4, probe_every=10, emb_std=0.02, train_frac=0.9, val_every=0,
        val_batches=20, probe_eval_mode=False, deterministic=False,
        optimizer="adam")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab, offset, batch_fn, _, _ = R.build_task(args)

    # 兩個固定且互相獨立的量測 batch(所有配置、所有時間點都用同一組)
    ga = torch.Generator().manual_seed(4242)
    gb = torch.Generator().manual_seed(8888)
    ba = tuple(t.to(dev) for t in batch_fn(ga))
    bb = tuple(t.to(dev) for t in batch_fn(gb))

    out = {}
    for key in cli.configs:
        cfg = R.CONFIG_BY_KEY[key]
        torch.manual_seed(cli.seed)
        model = TransformerLM(vocab, args.seq_len, args.d_model, args.n_heads,
                              args.d_ff, args.layers, cfg.norm, cfg.placement,
                              cond_detach=cfg.cond_detach,
                              emb_std=args.emb_std).to(dev)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        gen = torch.Generator().manual_seed(1234 + cli.seed)

        steps, cohs, mags, losses = [], [], [], []
        for step in range(0, args.steps + 1):
            if step % cli.every == 0:
                c, m = coherence(model, ba, bb, vocab)
                steps.append(step); cohs.append(c); mags.append(m)
            if step == args.steps:
                break
            x, y = batch_fn(gen)
            x, y = x.to(dev), y.to(dev)
            model.zero_grad(set_to_none=True)
            logits, _, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1),
                                   ignore_index=-100)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        out[key] = {"steps": steps, "coherence": cohs, "grad_rms": mags,
                    "losses": losses,
                    "final_loss": float(np.mean(losses[-10:]))}
        c = np.array(cohs)
        print(f"[{key:12s}] final_loss={out[key]['final_loss']:.4f}  "
              f"底 4 層 coherence 中位數={np.nanmedian(c[:, :4]):+.4f}  "
              f"頂 4 層={np.nanmedian(c[:, -4:]):+.4f}", flush=True)

    path = BASE / cli.out
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n輸出 {path}")


if __name__ == "__main__":
    main()
