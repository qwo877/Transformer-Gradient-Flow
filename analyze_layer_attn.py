# -*- coding: utf-8 -*-
import json
import sys
from pathlib import Path
from statistics import median, mean, stdev

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = Path(__file__).parent / "results"
NEW = BASE / "layer_attn_n15"
OLD = BASE / "reinject_n15_paired"
KEYS = ["ln_post", "post_reinject", "post_reinject_detach", "post_reinject_mlp",
        "post_reinject_tok", "pre_ln", "deepnorm"]
REINJ = ["post_reinject", "post_reinject_detach", "post_reinject_mlp", "post_reinject_tok"]
THR, THR_SENS = 0.15, (0.12, 0.20)
TOP = 3


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def wnorm_report(old):
    print("== 重注入權重 Frobenius 範數,依深度(reinject_n15_paired,15 顆種子)==")
    lines = []
    for k in REINJ:
        W = old[k]["reinject_wnorm_seeds"]           # 15 x 32
        n_pt = len(W[0])
        med = [median(s[i] for s in W) for i in range(n_pt)]
        bottom, top = med[: n_pt // 4], med[-n_pt // 4:]
        # 逐種子:頂部 1/4 的中位數是否大於底部 1/4 的中位數
        grows = sum(1 for s in W if median(s[-n_pt // 4:]) > median(s[: n_pt // 4]))
        argmax = [s.index(max(s)) + 1 for s in W]
        row = (f"{k:22s} 底部 8 點中位數 {median(bottom):.2f}  頂部 8 點 {median(top):.2f}  "
               f"比值 {median(top) / median(bottom):.1f}x  頂部>底部 {grows}/{len(W)} 顆  "
               f"最大值所在 norm 點 {min(argmax)}–{max(argmax)}(共 {n_pt} 點)")
        print(row)
        lines.append(row)
    return lines


def layer_report(m, old):
    out = []
    say = lambda s="": (print(s), out.append(s))

    say("== 重現性:final loss 逐種子與 reinject_n15_paired 比對 ==")
    for k in KEYS:
        if k not in m or k not in old:
            say(f"{k:22s} (缺資料)")
            continue
        a, b = m[k]["final_loss_seeds"], old[k]["final_loss_seeds"]
        d = max(abs(x - y) for x, y in zip(a, b))
        am = max(abs(x - y) for x, y in zip(m[k]["attn_prev_final_seeds"],
                                           old[k]["attn_prev_final_seeds"]))
        say(f"{k:22s} final loss 最大差 {d:.2e}   attn(層最大) 最大差 {am:.2e}")

    say("")
    say(f"== 逐層 t-1 attention(head max),15 顆種子平均;形成門檻 {THR} ==")
    for k in KEYS:
        if k not in m:
            continue
        A = m[k]["attn_prev_final_layers_seeds"]
        I = m[k]["attn_prev_init_layers_seeds"]
        L = len(A[0])
        prof = " ".join(f"{mean(s[l] for s in A):.2f}" for l in range(L))
        say(f"{k:22s} {prof}")

    say("")
    say("== 每顆種子    形成 的層(1-indexed)與頂部集中度 ==")
    summary = {}
    for k in KEYS:
        if k not in m:
            continue
        A = m[k]["attn_prev_final_layers_seeds"]
        I = m[k]["attn_prev_init_layers_seeds"]
        H = m[k].get("attn_prev_final_headmean_layers_seeds")
        L = len(A[0])
        rows, lows, tops_share, tops_share_hm, argmaxes = [], [], [], [], []
        within_top = {t: 0 for t in (THR,) + THR_SENS}
        any_formed = 0
        for si, (a, i) in enumerate(zip(A, I)):
            formed = [l + 1 for l in range(L) if a[l] >= THR]
            if formed:
                any_formed += 1
                lows.append(min(formed))
            for t in within_top:
                f = [l + 1 for l in range(L) if a[l] >= t]
                if f and min(f) > L - TOP:
                    within_top[t] += 1
            gain = [max(a[l] - i[l], 0.0) for l in range(L)]
            tot = sum(gain)
            tops_share.append(sum(gain[-TOP:]) / tot if tot > 0 else float("nan"))
            if H:
                h = H[si]
                gh = [max(h[l] - i[l], 0.0) for l in range(L)]   # head-mean 的 init 用 head-max 近似,只當參考
                th = sum(gh)
                tops_share_hm.append(sum(gh[-TOP:]) / th if th > 0 else float("nan"))
            argmaxes.append(a.index(max(a)) + 1)
            rows.append(formed)
        share = [x for x in tops_share if x == x]
        say(f"{k:22s} 有形成的種子 {any_formed}/{len(A)}  "
            f"最低形成層 {('%d–%d' % (min(lows), max(lows))) if lows else '  '}  "
            f"最大值所在層 {min(argmaxes)}–{max(argmaxes)}  "
            f"頂部{TOP}層佔增量 中位數 {median(share):.2f}(最小 {min(share):.2f})"
            if share else
            f"{k:22s} 有形成的種子 {any_formed}/{len(A)}  (沒有高於初始值的 attention)")
        sens = "  ".join(f"門檻 {t}: 全在頂部{TOP}層 {c}/{len(A)}" for t, c in within_top.items())
        say(f"{'':22s} {sens}")
        summary[k] = {"formed_layers_per_seed": rows, "lowest_formed": lows,
                      "argmax_layer": argmaxes, "top3_share": tops_share,
                      "within_top3": within_top, "any_formed": any_formed}

    # 門檻 0.15 的    全在頂部 3 層 對低層 0.15–0.17 的小凸起太敏感,不適合寫進論文(9/16)
    say("")
    say("== 門檻無關的集中度(逐種子;括號內是 15 顆的範圍)==")
    for k in KEYS:
        if k not in m or k == "ln_post":
            continue
        A = m[k]["attn_prev_final_layers_seeds"]
        I = m[k]["attn_prev_init_layers_seeds"]
        com, t4, t5, b8, amax = [], [], [], [], []
        for a, i in zip(A, I):
            gain = [max(x - y, 0.0) for x, y in zip(a, i)]
            tot = sum(gain)
            com.append(sum((l + 1) * g for l, g in enumerate(gain)) / tot)
            t4.append(sum(gain[-4:]) / tot)
            t5.append(sum(gain[-5:]) / tot)
            b8.append(sum(gain[:8]) / tot)
            amax.append(a.index(max(a)) + 1)
        rng = lambda v, f=".2f": f"{median(v):{f}}({min(v):{f}}–{max(v):{f}})"
        say(f"{k:22s} 最強層 {min(amax)}–{max(amax)}  重心 {rng(com, '.1f')}  "
            f"頂部4層 {rng(t4)}  頂部5層 {rng(t5)}  底部8層 {rng(b8)}")
        summary[k].update({"center_of_mass": com, "top4_share": t4,
                           "top5_share": t5, "bottom8_share": b8})
    return out, summary


def main():
    old = load(OLD / "metrics.json")
    wlines = wnorm_report(old)
    if "--wnorm-only" in sys.argv:
        return
    m = {}
    for k in KEYS:
        p = NEW / k / "metrics.json"
        if p.exists():
            m.update(load(p))
        else:
            print(f"(還沒有 {p})")
    if not m:
        return
    print()
    lines, summary = layer_report(m, old)
    md = ["# 逐層 t-1 attention 分析", "",
          "由 `analyze_layer_attn.py` 產生;資料 `results/layer_attn_n15/<key>/metrics.json`", "",
          "```", *wlines, "", *lines, "```", ""]
    (NEW / "summary_layer_attn.md").write_text("\n".join(md), encoding="utf-8")
    with open(NEW / "summary_layer_attn.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"\n已寫出 {NEW / 'summary_layer_attn.md'}")


if __name__ == "__main__":
    main()
