"""把 README.md / 研究報告.md 裡的每一個關鍵數字,對回 results/ 重算
用法:
    python verify_doc_numbers.py           # 全部
    python verify_doc_numbers.py -v        # 連通過的項目也列出
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

for _s in (sys.stdout, sys.stderr):            # 重導向輸出時 Windows 會退回 cp950,強制 utf-8
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).parent
FAIL = []
PASSED = []

#§

def load(name):
    p = BASE / "results" / name / "metrics.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def chk(label, got, exp, tol):
    ok = got is not None and abs(got - exp) <= tol
    (PASSED if ok else FAIL).append((label, got, exp))
    return ok


def med(arr, idx, lo=None, hi=None):
    a = np.array(arr)[idx]
    return float(np.nanmedian(a[:, lo:hi] if (lo, hi) != (None, None) else a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    cli = ap.parse_args()

    copy, modk, char = load("copy"), load("modk"), load("char_lm")

    #  §5.1 主表(copy):block ratio / 最終 loss / attention 
    for k, r, l, a in [
            ("ln_post", 1.82, 4.1619, 0.064), ("rms", 1.88, 4.1618, 0.064),
            ("adain", 37.52, 4.1494, 0.072), ("adain_loc", 37.52, 0.0088, 0.294),
            ("gn", 2.22, 3.79, 0.156), ("in", 38.66, 3.58, 0.127),
            ("bn", 3.21, 3.46, 0.154), ("spade", 32.72, 1.11, 0.289),
            ("pre_ln", 4.71, 0.0003, 0.606), ("sandwich", 3.65, 0.0003, 0.544),
            ("deepnorm", 0.99, 0.0005, 0.913), ("pre_rms", 4.58, 0.0003, 0.600),
            ("adain_loc_detach", 37.52, 1.40, 0.252),
            ("spade_detach", 32.72, 2.78, 0.147)]:
        chk(f"§5.1 {k} block ratio", float(np.mean(copy[k]["init_ratio_block_seeds"])), r, 6e-3)
        chk(f"§5.1 {k} final loss", float(np.mean(copy[k]["final_loss_seeds"])), l, 6e-3)
        chk(f"§5.1 {k} attn", float(np.mean(copy[k]["attn_prev_final_seeds"])), a, 1e-3)

    #  §5.2 跨任務表:copy / modk / char_lm(train + val
    for k, cp, mk, tr, va in [
            ("ln_post", 4.1619, 4.1620, 3.3178, 3.3469), ("rms", 4.1618, 4.1620, 3.3227, 3.3469),
            ("bn", 3.46, 4.1604, 2.7710, 2.8050), ("gn", 3.79, 4.1590, 1.7004, 1.7165),
            ("in", 3.58, 4.1592, 0.3078, 0.3145), ("adain", 4.1494, 4.1595, 0.2830, 0.2904),
            ("adain_loc", 0.0088, 4.1592, 0.2053, 0.2339), ("spade", 1.11, 4.1593, 0.2497, 0.2683),
            ("wn", 4.077, 4.1719, 2.5834, 2.5905), ("pre_ln", 0.0003, 2.93, 1.6563, 1.8346),
            ("sandwich", 0.0003, 0.71, 1.6115, 1.7965),
            ("deepnorm", 0.0005, 4.1618, 1.7151, 1.8839),
            ("pre_rms", 0.0003, 2.85, 1.6632, 1.8405)]:
        chk(f"§5.2 {k} copy", float(np.mean(copy[k]["final_loss_seeds"])), cp, 6e-3)
        chk(f"§5.2 {k} modk", float(np.mean(modk[k]["final_loss_seeds"])), mk, 6e-3)
        chk(f"§5.2 {k} char train", float(np.mean(char[k]["final_loss_seeds"])), tr, 6e-4)
        chk(f"§5.2 {k} char val", float(np.mean(char[k]["val_loss_seeds"])), va, 6e-4)

    #  held-out n-gram baseline 
    b = char["ln_post"]["ngram_baselines"]
    for n, e in [("uniform", 4.1744), ("unigram", 3.3473), ("bigram", 2.4819), ("trigram", 2.0697)]:
        chk(f"baseline {n}", float(b[n]), e, 1e-3)

    #  H3 三方 ablation(15 種子)與穩健性(10 種子
    n15, stab = load("h3_n15"), load("stability")
    for k, e in [("adain", 0), ("adain_loc", 9), ("adain_loc_detach", 9)]:
        g = sum(1 for v in n15[k]["final_loss_seeds"] if v < 0.1)
        chk(f"h3_n15 {k} 解掉數", float(g), float(e), 0)
    for k, e in [("bn", 1), ("gn", 0), ("in", 0), ("spade", 2)]:
        g = sum(1 for v in stab[k]["final_loss_seeds"] if v < 0.1)
        chk(f"stability {k} 解掉數", float(g), float(e), 0)

    #  2x2 解耦 與 lr 掃描 
    for tag, vals in [
            ("decouple2_adam", {"ln_post": 4.1619, "post_ln_gr": 4.1630, "pre_ln": 0.0003}),
            ("decouple2_sgd0.01", {"ln_post": 4.1619, "post_ln_gr": 4.1626, "pre_ln": 0.0018}),
            ("decouple2_sgd0.03", {"ln_post": 4.1642, "post_ln_gr": 4.1645, "pre_ln": 0.0006})]:
        d = load(tag)
        for k, e in vals.items():
            chk(f"{tag} {k}", float(np.mean(d[k]["final_loss_seeds"])), e, 6e-4)
    for lr, vals in [("1e-4", {"ln_post": 1.1956, "pre_ln": 0.0011, "adain_loc": 2.7934,
                               "post_ln_st": 4.1805}),
                     ("1e-3", {"ln_post": 4.1633, "pre_ln": 0.0001, "adain_loc": 0.0011,
                               "post_ln_st": 4.1792})]:
        d = load(f"copy_lr{lr}")
        for k, e in vals.items():
            chk(f"lr={lr} {k}", float(np.mean(d[k]["final_loss_seeds"])), e, 6e-4)

    #  §7-6 補充量測:coherence(cosine)與 grad_rms 
    # grad_rms 就是 F2 漏掉的那一組
    coh = json.load(open(BASE / "results" / "grad_coherence.json", encoding="utf-8"))
    for k, (lo, mid, hi) in {"ln_post": (-0.22, -0.10, 0.16),
                             "post_ln_gr": (0.09, 0.12, 0.20),
                             "pre_ln": (0.21, 0.32, 0.62)}.items():
        idx = [i for i, s in enumerate(coh[k]["steps"]) if s >= 700]
        c = coh[k]["coherence"]
        chk(f"coherence {k} 底4層", med(c, idx, None, 4), lo, 5e-3)
        chk(f"coherence {k} 中8層", med(c, idx, 4, 12), mid, 5e-3)
        chk(f"coherence {k} 頂4層", med(c, idx, -4, None), hi, 5e-3)
    for k, e in [("pre_ln", 3.66e-07), ("post_ln_gr", 1.12e-03)]:
        idx = [i for i, s in enumerate(coh[k]["steps"]) if s >= 700]
        chk(f"grad_rms {k} 底4層(F2 漏掉的那組)", med(coh[k]["grad_rms"], idx, None, 4),
            e, e * 0.02)

    #  A5:初始化時 activation 與參數梯度的底/頂,符號相反 
    for k, ea, ep in [("ln_post", 1.82, 0.69), ("pre_ln", 4.71, 2.50)]:
        act = float(np.mean(copy[k]["init_ratio_block_seeds"]))
        g = np.array(coh[k]["grad_rms"][0])
        par = float(np.median(g[:4]) / np.median(g[-4:]))
        chk(f"A5 {k} activation 底/頂", act, ea, 6e-3)
        chk(f"A5 {k} 參數梯度 底/頂", par, ep, 1e-2)


    rj = load("reinject_n15")
    for k, n_esc, m in [("ln_post", 0, 4.1621), ("adain_loc", 9, 0.0258),
                        ("post_reinject", 15, 0.0024),
                        ("post_reinject_detach", 15, 0.0045)]:
        L = rj[k]["final_loss_seeds"]
        chk(f"實驗1 {k} 解掉數", float(sum(1 for v in L if v < 0.1)), float(n_esc), 0)
        chk(f"實驗1 {k} 中位數", float(np.median(L)), m, 6e-4)
    chk("實驗1 post_reinject 參數量", float(rj["post_reinject"]["n_params"]),
        3725376.0, 0)
    chk("實驗1 adain_loc 參數量", float(rj["adain_loc"]["n_params"]), 4245568.0, 0)
    # G4:重注入權重範數(zero-init 起點 = 0)訓練後必須明顯離開 0
    _wn = [v for row in rj["post_reinject"]["reinject_wnorm_seeds"] for v in row]
    chk("實驗1 G4 重注入權重範數最小值", float(min(_wn)), 0.3308, 5e-4)
    chk("實驗1 G4 重注入權重範數中位數", float(np.median(_wn)), 1.6356, 5e-4)
    # cond/trunk 佔比:這是「為什麼非補 detach 對照不可」的那個數字
    chk("實驗1 post_reinject cond/trunk(seed0)",
        rj["post_reinject"]["final_cond_rms"] / rj["post_reinject"]["final_trunk_rms"],
        172.63, 0.05)
    chk("實驗1 adain_loc cond/trunk(seed0)",
        rj["adain_loc"]["final_cond_rms"] / rj["adain_loc"]["final_trunk_rms"],
        6.34, 5e-3)
    for k, a in [("post_reinject", 0.230), ("post_reinject_detach", 0.231),
                 ("adain_loc", 0.272)]:
        chk(f"實驗1 {k} attn",
            float(np.mean(rj[k]["attn_prev_final_seeds"])), a, 1e-3)

    g4d = load("gr_lr1e-4_n15")
    for k, mean, medv in [("ln_post", 1.6999, 1.0590),
                          ("post_ln_gr", 1.1989, 1.0200)]:
        L = g4d[k]["final_loss_seeds"]
        chk(f"實驗2 {k} 平均", float(np.mean(L)), mean, 6e-4)
        chk(f"實驗2 {k} 中位數", float(np.median(L)), medv, 6e-4)
    chk("實驗2 ln_post 最小值", float(np.min(g4d["ln_post"]["final_loss_seeds"])),
        0.8241, 6e-4)
    chk("實驗2 ln_post 最大值", float(np.max(g4d["ln_post"]["final_loss_seeds"])),
        3.7003, 6e-4)

    for tag, vals in [("healthy_ctrl_pre",
                       {"pre_ln": 0.0003, "pre_ln_gr": 0.0009}),
                      ("healthy_ctrl_deepnorm",
                       {"deepnorm": 0.0005, "deepnorm_gr": 0.0000})]:
        d = load(tag)
        for k, e in vals.items():
            chk(f"實驗3 {tag} {k} 平均",
                float(np.mean(d[k]["final_loss_seeds"])), e, 6e-5)
    for tag, k, a in [("healthy_ctrl_pre", "pre_ln", 0.606),
                      ("healthy_ctrl_pre", "pre_ln_gr", 0.975),
                      ("healthy_ctrl_deepnorm", "deepnorm", 0.913),
                      ("healthy_ctrl_deepnorm", "deepnorm_gr", 0.998)]:
        chk(f"實驗3 {tag} {k} attn",
            float(np.mean(load(tag)[k]["attn_prev_final_seeds"])), a, 1e-3)
    # 校準值:文件引用的是 5 位有效數字 與 calibration.json 的全精度值一致
    cal = json.load(open(BASE / "results" / "calibration.json", encoding="utf-8"))
    for k, e in [("ln_post", 6.0073e-05), ("pre_ln", 1.4881e-05),
                 ("deepnorm", 5.0369e-05)]:
        chk(f"校準 {k} target_rms", cal[k]["target_rms_seed0"], e, e * 1e-4)

    #  實驗 4(lr=1e-4 的 detach)與 實驗 5(2x2 解耦擴種子)
    d4 = load("detach_lr1e-4_n15")
    for k, n_esc, m in [("adain_loc", 0, 4.1523), ("adain_loc_detach", 0, 4.0488)]:
        L = d4[k]["final_loss_seeds"]
        chk(f"實驗4 {k} 解掉數", float(sum(1 for v in L if v < 0.1)), float(n_esc), 0)
        chk(f"實驗4 {k} 中位數", float(np.median(L)), m, 6e-4)
    # 最好的那顆種子離 <0.1 判準只差 0.003  §5.8「判準鬆緊會改變結論」的又一例
    chk("實驗4 adain_loc 最佳種子", float(np.min(d4["adain_loc"]["final_loss_seeds"])),
        0.1032, 6e-4)
    chk("實驗4 adain_loc_detach 最佳種子",
        float(np.min(d4["adain_loc_detach"]["final_loss_seeds"])), 0.1267, 6e-4)

    for tag, vals in [
            ("decouple2_adam_n10",
             {"ln_post": 4.1618, "post_ln_gr": 4.1621, "pre_ln": 0.0003}),
            ("decouple2_sgd0.01_n10",
             {"ln_post": 4.1618, "post_ln_gr": 4.1623, "pre_ln": 0.0018})]:
        d = load(tag)
        for k, e in vals.items():
            L = d[k]["final_loss_seeds"]
            chk(f"實驗5 {tag} {k} 平均", float(np.mean(L)), e, 6e-5)
            chk(f"實驗5 {tag} {k} 種子數", float(len(L)), 10.0, 0)
        # 通過條件之一:兩個 Post 配置的每一顆種子都在隨機水準 ±0.05 內
        for k in ("ln_post", "post_ln_gr"):
            worst = float(np.max(np.abs(np.array(d[k]["final_loss_seeds"])
                                        - np.log(64))))
            chk(f"實驗5 {tag} {k} 距隨機水準最大偏離", worst, 0.0, 0.05)

    #  README 引用的統計檢定與衍生量 
    # p 值也是 公布的數字 同樣要能對回原始資料重算 不能只存在 analysis.json
    def _fisher(a_g, a_n, b_g, b_n):
        return float(stats.fisher_exact([[a_g, a_n - a_g], [b_g, b_n - b_g]],
                                        alternative="two-sided")[1])

    def _esc(d, k):
        L = d[k]["final_loss_seeds"]
        return sum(1 for v in L if v < 0.1), len(L)

    _n15 = load("h3_n15")
    chk("p 值 adain vs adain_loc(H3 granularity)",
        _fisher(*_esc(_n15, "adain"), *_esc(_n15, "adain_loc")), 0.0007, 5e-5)
    chk("p 值 adain_loc vs detach(H3 bypass 非必要)",
        _fisher(*_esc(_n15, "adain_loc"), *_esc(_n15, "adain_loc_detach")), 1.0, 1e-9)
    chk("p 值 post_reinject vs adain_loc(實驗1)",
        _fisher(*_esc(rj, "post_reinject"), *_esc(_n15, "adain_loc")), 0.0169, 5e-5)
    chk("p 值 post_reinject vs 其 detach 版(實驗1 追加)",
        _fisher(*_esc(rj, "post_reinject"), *_esc(rj, "post_reinject_detach")),
        1.0, 1e-9)
    chk("p 值 實驗2 Mann-Whitney(lr=1e-4)",
        float(stats.mannwhitneyu(g4d["post_ln_gr"]["final_loss_seeds"],
                                 g4d["ln_post"]["final_loss_seeds"],
                                 alternative="two-sided")[1]), 0.4553, 5e-5)
    d4b = load("detach_lr1e-4_n15")
    chk("p 值 實驗4 Fisher(lr=1e-4 detach)",
        _fisher(*_esc(d4b, "adain_loc"), *_esc(d4b, "adain_loc_detach")), 1.0, 1e-9)

    # DeepNorm 加干預後變好 3.21 個數量級
    _hd = load("healthy_ctrl_deepnorm")
    chk("實驗3 DeepNorm 干預後的數量級變化",
        float(np.log10(np.mean(_hd["deepnorm_gr"]["final_loss_seeds"])
                       / np.mean(_hd["deepnorm"]["final_loss_seeds"]))), -3.21, 5e-3)

    # 每個 norm 點的條件模組參數量(README 主表引用):
    # AdaIN-local 的 to_gamma+to_beta 是 2*(d*d+d);變體 A 是它的一半
    _d = 128
    chk("條件模組參數/norm 點 AdaIN-local", float(2 * (_d * _d + _d)), 33024.0, 0)
    chk("條件模組參數/norm 點 純重注入", float(_d * _d + _d), 16512.0, 0)

    # H4 的 attention 分佈
    _sol, _stk = [], []
    for k in ("adain", "adain_loc", "adain_loc_detach"):
        for l, a in zip(_n15[k]["final_loss_seeds"], _n15[k]["attn_prev_final_seeds"]):
            (_sol if l < 0.1 else _stk).append(a)
    chk("H4 解掉數(h3_n15)", float(len(_sol)), 18.0, 0)
    chk("H4 卡死數(h3_n15)", float(len(_stk)), 27.0, 0)
    chk("H4 解掉者 attention 最小", float(min(_sol)), 0.237, 1e-3)
    chk("H4 解掉者 attention 最大", float(max(_sol)), 0.573, 1e-3)
    chk("H4 卡死者 attention 最大", float(max(_stk)), 0.340, 1e-3)
    chk("H4 兩分佈重疊(卡死max > 解掉min)",
        float(max(_stk) > min(_sol)), 1.0, 0)

    # ---- 報告 ----
    print("=" * 72)
    print(f"文件數字核對:{len(PASSED)} 通過 / {len(FAIL)} 不符(共 {len(PASSED)+len(FAIL)} 項)")
    print("=" * 72)
    if cli.verbose:
        for lbl, got, exp in PASSED:
            print(f"  [OK] {lbl:44s} {got:>12.4g}  文件:{exp}")
    for lbl, got, exp in FAIL:
        print(f"  [X ] {lbl:44s} {got:>12.4g}  文件:{exp}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
