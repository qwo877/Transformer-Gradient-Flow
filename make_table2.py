# -*- coding: utf-8 -*-
#從 metrics.json 產生論文的表 2(Markdown + LaTeX)
import json
import sys
from pathlib import Path

from statistics import stdev

from scipy.stats import beta, binomtest, fisher_exact

BASE = Path(__file__).parent / "results"
ESCAPE = 0.1
BASELINE = "ln_post"
ROWS = ["ln_post", "adain_loc", "post_reinject",
        "post_reinject_detach", "post_reinject_mlp"]

LABEL = {
    "ln_post":              ("Post-LN",        None),
    "adain_loc":            ("AdaIN-local",    33024),
    "post_reinject":        ("Re-inj.",        16512),
    "post_reinject_detach": ("Re-inj.+detach", 16512),
    "post_reinject_mlp":    ("Re-inj. (MLP)",  33024),
}

# E2:cond 的成分分解 第二欄改成 cond 的來源,參數量對這幾列全部相同(16,512)沒有資訊
E2_ROWS = ["ln_post", "post_reinject", "post_reinject_tok",
           "post_reinject_pos", "post_reinject_rand"]
E2_LABEL = {
    "ln_post":            ("Post-LN (baseline)",            "  "),
    "post_reinject":      ("re-injection, full cond",       "tok + pos(內容 + 位置,可學)"),
    "post_reinject_tok":  ("re-injection, token only",      "tok(只有內容,可學)"),
    "post_reinject_pos":  ("re-injection, position only",   "pos(只有位置,可學)"),
    "post_reinject_rand": ("re-injection, frozen random",   "rand(只有位置區分度,凍結)"),
}
MCNEMAR_FLOOR = 2 * 0.5 ** 15          # 15 對種子的 p 值下限


def clopper_pearson(k, n, alpha=0.05):
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k else 0.0
    hi = beta.isf(alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def mean(xs):
    return sum(xs) / len(xs)


def escape_step(losses):
    return next((i + 1 for i, v in enumerate(losses) if v < ESCAPE), None)


def collect(m, keys=None):
    base_solved = [v < ESCAPE for v in m[BASELINE]["final_loss_seeds"]]
    rows = []
    for key in (keys or ROWS):
        r = m[key]
        loss = r["final_loss_seeds"]
        n = len(loss)
        solved = [v < ESCAPE for v in loss]
        k = sum(solved)
        steps = [s for s in (escape_step(ls) for ls in r["losses_all"]) if s]
        # McNemar:只看不一致對 b = 只有 baseline 解掉 c = 只有本列解掉
        b = sum(1 for a, c in zip(base_solved, solved) if a and not c)
        c = sum(1 for a, d in zip(base_solved, solved) if not a and d)
        if key == BASELINE:
            p_mc = p_f = None
        else:
            p_mc = binomtest(c, b + c, 0.5).pvalue if (b + c) else 1.0
            p_f = fisher_exact([[k, n - k],
                                [sum(base_solved), n - sum(base_solved)]])[1]
        rows.append({
            "key": key, "k": k, "n": n, "ci": clopper_pearson(k, n),
            "p_mc": p_mc, "p_fisher": p_f, "b": b, "c": c,
            "loss_med": median(loss),
            "step_med": median(steps) if steps else None,
            "attn": mean(r["attn_prev_final_seeds"]),
            "attn_sd": stdev(r["attn_prev_final_seeds"]),
            "attn_init": mean(r["attn_prev_init_seeds"]),
            "attn_hm": (mean(r["attn_prev_final_headmean_seeds"])
                        if r.get("attn_prev_final_headmean_seeds") else None),
        })
    return rows


def p_plain(p):
    #給 MD / 終端機看的 p 值
    if p is None:
        return "  "
    return f"{p:.3g}"


def p_tex(p):
    #給 LaTeX 看的 p 值  $a\\times10^{b}$
    if p is None:
        return "--"
    if p >= 1e-3:
        return f"${p:.4f}$"
    mant, exp = f"{p:.1e}".split("e")
    return f"${mant}\\times10^{{{int(exp)}}}$"


def render_markdown(rows, labels=None, col2="條件參數<br>/norm 點"):
    labels = labels or LABEL
    out = [f"| 配置 | {col2} | 解掉 | 95% CP CI | McNemar $p$ "
           "| loss 中位數 | 逃脫步數<br>中位數 | attn→t−1<br>(init) |",
           "|---|---:|---:|:---:|:---:|---:|---:|---:|"]
    for r in rows:
        par = labels[r["key"]][1]
        out.append(
            f"| `{r['key']}` "
            f"| {'  ' if par is None else (par if isinstance(par, str) else format(par, ','))} "
            f"| **{r['k']}/{r['n']}** "
            f"| [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}] "
            f"| {p_plain(r['p_mc'])} "
            f"| {r['loss_med']:.4f} "
            f"| {'  ' if r['step_med'] is None else format(r['step_med'], '.0f')} "
            f"| {r['attn']:.3f} ± {r['attn_sd']:.3f} "
            f"({r['attn_init']:.3f}) |")
    return "\n".join(out)


def render_latex(rows):
    head = [
        r"% 需要 \usepackage{booktabs}",
        r"\begin{table}[tb]",
        r"\centering",
        
        r"\caption{Escape rates on the copy task (15 paired seeds; "
        r"Clopper--Pearson 95\% CIs); attn$\to t{-}1$ is the largest attention weight "
        r"on $t-1$ over layers and heads.}",
        r"\label{tab:main}",
        r"\smallskip",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",  
        r"\begin{tabular}{lrrcrrc}",
        r"\toprule",
        r"Configuration & \multicolumn{1}{c}{\begin{tabular}{@{}r@{}}Cond.\ par.\\"
        r"/norm pt.\end{tabular}} & Escape & 95\% CI & "
        r"\multicolumn{1}{c}{\begin{tabular}{@{}c@{}}McNemar\\$p$\end{tabular}} & "
        r"\multicolumn{1}{c}{\begin{tabular}{@{}r@{}}Median\\loss\end{tabular}} & "
        
        r"\multicolumn{1}{c}{\begin{tabular}{@{}c@{}}attn$\to t{-}1$\\"
        r"(init $\approx 0.07$)\end{tabular}} \\",
        r"\midrule",
    ]
    init_vals = [r["attn_init"] for r in rows]
    assert max(init_vals) - min(init_vals) < 0.01, \
        f"初始化 attention 各列差異變大了({min(init_vals):.3f}-{max(init_vals):.3f})," \
        f"欄頭的    init ≈ 0.07 不再成立,要改回逐列顯示"
    body = []
    for r in rows:
        label, par = LABEL[r["key"]]
        esc = f"{r['k']}/{r['n']}"
        if r["k"] == r["n"]:
            esc = r"\textbf{" + esc + "}"
        body.append(
            f"{label} & "
            f"{'--' if par is None else format(par, ',').replace(',', '{,}')} & "
            f"{esc} & "
            f"[{r['ci'][0]:.2f},\\,{r['ci'][1]:.2f}] & "
            f"{p_tex(r['p_mc'])} & "
            f"{r['loss_med']:.4f} & "
            # 三位小數:兩位會讓 baseline 的標準差變成 0.00
            # 標準差要留:AdaIN-local 的 ±0.150 就是雙峰的證據(H4 必要非充分)
            f"${r['attn']:.3f}{{\\pm}}{r['attn_sd']:.3f}$ \\\\")
    return "\n".join(head + body + [r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def main(*argv):
    try:                    # Windows 主控台預設 cp950,編不了 U+2212 之類的字元
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    e2 = "--e2" in argv
    pos = [a for a in argv if not a.startswith("--")]
    out_dir = pos[0] if pos else "reinject_n15_paired"
    d = BASE / out_dir
    m = json.load(open(d / "metrics.json", encoding="utf-8"))

    if e2:
        missing = [k for k in E2_ROWS if k not in m]
        if missing:
            sys.exit(f"{d} 裡缺少 E2 的配置:{missing}(先把 E2 跑完)")
        rows = collect(m, E2_ROWS)
        md = render_markdown(rows, E2_LABEL, "cond 來源")
        (d / "table_e2.md").write_text(md + "\n", encoding="utf-8")
        print(md, "\n")
        print("-" * 78)
        print("\nE2 這張表不進論文(正文只用 1–2 句話帶過),所以只出 Markdown")
        print("\nMcNemar 不一致對(b = 只有 baseline 解掉,c = 只有該列解掉):")
        for r in rows[1:]:
            print(f"  {r['key']:22s} b={r['b']:2d} c={r['c']:2d}   "
                  f"McNemar {r['p_mc']:.3g}")
        print(f"\n已寫出 {d / 'table_e2.md'}")
        return

    rows = collect(m)
    md, tex = render_markdown(rows), render_latex(rows)
    # 先落檔再列印:主控台編碼出問題也不會弄丟輸出
    (d / "table2.md").write_text(md + "\n", encoding="utf-8")
    (d / "table2.tex").write_text(tex + "\n", encoding="utf-8")
    print(md, "\n")
    print("-" * 78, "\n")
    print(tex, "\n")
    print("-" * 78)

    print("\nattention 口徑對照(head_max 進表,head_mean 是舊口徑):")
    for r in rows:
        if r["attn_hm"] is not None:
            print(f"  {r['key']:22s} head_max {r['attn']:.3f}"
                  f"   head_mean {r['attn_hm']:.3f}"
                  f"   稀釋倍數 {r['attn'] / max(r['attn_hm'], 1e-9):.2f}x")
    base_analytic = m[BASELINE].get("attn_baseline")
    print(f"  單層解析 baseline = {base_analytic:.3f},但表裡用的是同一統計量在"
          f"初始化時的實測值(取最大有選擇偏差)")

    print("\nMcNemar 不一致對(b = 只有 baseline 解掉,c = 只有該列解掉):")
    for r in rows[1:]:
        floor = "  ← 已達 15 對的 p 值下限" if abs(
            r["p_mc"] - MCNEMAR_FLOOR) < 1e-12 else ""
        print(f"  {r['key']:22s} b={r['b']:2d} c={r['c']:2d}   "
              f"McNemar {r['p_mc']:.3g}   (Fisher 若誤用會給 {r['p_fisher']:.3g})"
              f"{floor}")

    print(f"\n已寫出 {d / 'table2.md'} 與 {d / 'table2.tex'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
