# -*- coding: utf-8 -*-
import sys

import torch

from run_experiment import CONFIG_BY_KEY
from model import TransformerLM

VOCAB, SEQ_LEN = 64, 64
BASELINE = "ln_post"
TARGETS = ["adain", "adain_loc", "adain_loc_detach", "spade", "spade_detach",
           "post_reinject", "post_reinject_detach", "post_reinject_mlp",
           # E2:三個只換 cond 來源的配置 重注入投影仍是 zero-init 
           # rand 的向量是 buffer 不進主幹 所以三者在 step 0 都該與 ln_post 相同
           "post_reinject_tok", "post_reinject_pos", "post_reinject_rand"]


def build(key, seed):
    cfg = CONFIG_BY_KEY[key]
    torch.manual_seed(seed)
    return TransformerLM(VOCAB, SEQ_LEN, norm_name=cfg.norm,
                         placement=cfg.placement,
                         cond_detach=cfg.cond_detach, reinject=cfg.reinject,
                         cond_source=cfg.cond_source)


def main(seeds=(0, 1, 2)):
    torch.manual_seed(999)
    probe = torch.randint(0, VOCAB, (4, SEQ_LEN))
    all_ok = True

    for key in TARGETS:
        rows = []
        for seed in seeds:
            base, cand = build(BASELINE, seed), build(key, seed)
            sa, sb = base.state_dict(), cand.state_dict()
            shared = [k for k in sa if k in sb]
            bad = [k for k in shared if not torch.equal(sa[k], sb[k])]
            extra = sum(p.numel() for p in cand.parameters()) \
                - sum(p.numel() for p in base.parameters())
            with torch.no_grad():
                dlogit = (base(probe)[0] - cand(probe)[0]).abs().max().item()
            rows.append((seed, len(shared), len(bad), extra, dlogit))

        ok = all(b == 0 for _, _, b, _, _ in rows)
        # AdaIN 系把 LayerNorm 換掉 共有張量本來就少 兩層意義要分開看:
        #   mismatched > 0  = RNG 串流被污染(這是 E1 要修的)
        #   dlogit    > 0   = 初始化時不是同一個函數(zero-init 應保證為 0)
        dl = max(r[4] for r in rows)
        all_ok &= ok
        print(f"{'PASS' if ok else 'FAIL'}  {key:22s} "
              f"shared={rows[0][1]:3d}  mismatched={max(r[2] for r in rows):3d}  "
              f"extra_params={rows[0][3]:+8d}  max|Δlogit|@init={dl:.3e}")

    print("\n所有配置與 ln_post 逐位元配對" if all_ok
          else "\n有配置未配對 見上方 FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
