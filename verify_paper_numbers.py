# -*- coding: utf-8 -*-
import json
import math
import sys
from pathlib import Path
from statistics import median, mean, stdev

import numpy as np
from scipy.stats import beta, binomtest

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = Path(__file__).parent / "results"
ESC = 0.1
FAILS, PASSES = [], []


def load(name):
    with open(BASE / name / "metrics.json", encoding="utf-8") as f:
        return json.load(f)


def check(where, claim, ok, got):
    (PASSES if ok else FAILS).append(where)
    print(f"  {'PASS' if ok else 'FAIL'}  [{where}] {claim}  (got: {got})")


def escapes(r, thr=ESC):
    return sum(v < thr for v in r["final_loss_seeds"])


def cp(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k else 0.0
    hi = beta.isf(0.025, k + 1, n - k) if k < n else 1.0
    return round(lo, 2), round(hi, 2)


def mcnemar(base, row, thr=ESC):
    b = sum(1 for x, y in zip(base["final_loss_seeds"], row["final_loss_seeds"]) if x < thr <= y)
    c = sum(1 for x, y in zip(base["final_loss_seeds"], row["final_loss_seeds"]) if y < thr <= x)
    return binomtest(c, b + c, 0.5).pvalue if b + c else 1.0


def first_below(curve, thr):
    idx = np.where(np.asarray(curve) < thr)[0]
    return int(idx[0]) + 1 if len(idx) else None


m = load("reinject_n15_paired")
post, reinj, det, mlp = m["ln_post"], m["post_reinject"], m["post_reinject_detach"], m["post_reinject_mlp"]
tok, pos, rand, adain, pre = (m["post_reinject_tok"], m["post_reinject_pos"], m["post_reinject_rand"],
                              m["adain_loc"], m["pre_ln"])
FLOOR = 2 * 0.5 ** 15

print("== Abstract / Introduction / Setup")
check("Abs/Intro", "Post-LN fails in all 15 seeds", escapes(post) == 0, f"{escapes(post)}/15")
check("Abs/Intro", "loss at chance level ln 64 = 4.159",
      abs(median(post["final_loss_seeds"]) - math.log(64)) < 0.01 and round(math.log(64), 3) == 4.159,
      f"median {median(post['final_loss_seeds']):.4f}, ln64 {math.log(64):.4f}")
check("Intro", "largest attention on t-1 stays near its baseline (final ≈ init)",
      abs(mean(post["attn_prev_final_seeds"]) - mean(post["attn_prev_init_seeds"])) < 0.01,
      f"{mean(post['attn_prev_final_seeds']):.3f} vs init {mean(post['attn_prev_init_seeds']):.3f}")
check("Abs", "re-injection rescues all 15 seeds", escapes(reinj) == 15, f"{escapes(reinj)}/15")
check("Abs", "detaching also rescues all 15 seeds", escapes(det) == 15, f"{escapes(det)}/15")
check("Setup", "Pre-LN escapes in 15/15 seeds", escapes(pre) == 15, f"{escapes(pre)}/15")
check("Setup", "Pre-LN median loss 0.0003", f"{median(pre['final_loss_seeds']):.4f}" == "0.0003",
      f"{median(pre['final_loss_seeds']):.4f}")
lo_ok = max(reinj["final_loss_seeds"] + det["final_loss_seeds"]) < 0.01
hi_ok = min(post["final_loss_seeds"]) > 2.0
check("Setup", "Post-LN 0/15 vs re-inj (with/without detach) 15/15 unchanged for thresholds 0.01–2.0",
      lo_ok and hi_ok,
      f"re-inj/detach max {max(reinj['final_loss_seeds'] + det['final_loss_seeds']):.4f}, "
      f"Post-LN min {min(post['final_loss_seeds']):.4f}")

prov = json.load(open(BASE / "reinject_n15_paired" / "provenance.json", encoding="utf-8"))
args = prov["shards"][0]["args"]
want = {"layers": 16, "d_model": 128, "n_heads": 4, "d_ff": 512, "vocab": 64, "seq_len": 64,
        "batch": 32, "lr": 3e-4, "steps": 1500, "seeds": 15, "optimizer": "adam"}
same = all(all(s["args"][k] == v for k, v in want.items()) for s in prov["shards"])
check("Setup", "16 layers, d=128, 4 heads, d_ff=512, vocab 64, length 64, batch 32, Adam 3e-4, 1,500 steps, 15 seeds",
      same, "all shards" if same else "mismatch")
src = (Path(__file__).parent / "run_experiment.py").read_text(encoding="utf-8")
check("Setup", "no warmup / LR schedule in the training loop",
      "lr_scheduler" not in src and "warmup" not in src.lower().replace("warmup 會遮住", ""),
      "no scheduler found")

print("== Method")
d = 128
n_reinj = reinj["n_params"] - post["n_params"]
check("Method/Table", "each d×d projection plus bias = 16,512 per norm point; 32 points",
      n_reinj == 32 * (d * d + d) == 528_384, f"added {n_reinj:,} = 32 × {n_reinj // 32:,}")
wn = det["reinject_wnorm_seeds"]
check("Method", "with detach, all 32 projections still learn (every norm > 0)",
      all(len(s) == 32 and min(s) > 0 for s in wn), f"min norm {min(min(s) for s in wn):.3f}")
check("Method/Results", "MLP projection: 33,024 conditional parameters per point",
      (mlp["n_params"] - post["n_params"]) // 32 == 2 * (d * d + d) == 33_024,
      f"{(mlp['n_params'] - post['n_params']) // 32:,}")

print("== Table 1")
rows = [("Post-LN", post, 0, (0.00, 0.22), None, "4.1621", (0.066, 0.002)),
        ("AdaIN-local", adain, 9, (0.32, 0.84), "0.0039", "0.0771", (0.265, 0.150)),
        ("Re-inj.", reinj, 15, (0.78, 1.00), "6.1e-05", "0.0025", (0.252, 0.012)),
        ("Re-inj.+detach", det, 15, (0.78, 1.00), "6.1e-05", "0.0051", (0.248, 0.005)),
        ("Re-inj. (MLP)", mlp, 15, (0.78, 1.00), "6.1e-05", "0.0025", (0.319, 0.055))]
for name, r, k, ci, p, med, att in rows:
    got_p = None if r is post else mcnemar(post, r)
    p_txt = None if got_p is None else (f"{got_p:.4f}" if got_p >= 1e-3 else f"{got_p:.1e}")
    a = r["attn_prev_final_seeds"]
    ok = (escapes(r) == k and cp(k, 15) == ci and p_txt == p
          and f"{median(r['final_loss_seeds']):.4f}" == med
          and (round(mean(a), 3), round(stdev(a), 3)) == att)
    check("Table 1", f"{name}: {k}/15, CI {ci}, McNemar {p}, median {med}, attn {att[0]}±{att[1]}", ok,
          f"{escapes(r)}/15, {cp(escapes(r), 15)}, {p_txt}, {median(r['final_loss_seeds']):.4f}, "
          f"{mean(a):.3f}±{stdev(a):.3f}")
inits = [mean(r["attn_prev_init_seeds"]) for _, r, *_ in rows]
check("Table 1", "attn column header: init ≈ 0.07", all(abs(x - 0.07) < 0.006 for x in inits),
      ", ".join(f"{x:.3f}" for x in inits))
check("Table 1", "adain_loc conditional parameters 33,024 per point (2 × 16,512)", 2 * (d * d + d) == 33_024, "33,024")

print("== Results")
check("Results", "McNemar p = 6.1e-5 is the smallest attainable for 15 pairs",
      abs(mcnemar(post, reinj) - FLOOR) < 1e-12 and f"{FLOOR:.1e}" == "6.1e-05", f"{FLOOR:.2e}")
diff01 = [first_below(b, 0.1) - first_below(a, 0.1) for a, b in zip(reinj["losses_all"], det["losses_all"])]
check("Results", "detach reaches loss 0.1 later on every paired seed by 176–294 steps",
      min(diff01) == 176 and max(diff01) == 294 and all(x > 0 for x in diff01), f"{min(diff01)}–{max(diff01)}")
all_pos = True
for thr in np.geomspace(0.05, 2.0, 400):
    ds = [first_below(b, thr) - first_below(a, thr) for a, b in zip(reinj["losses_all"], det["losses_all"])]
    all_pos &= all(x > 0 for x in ds)
check("Results", "sign test p = 6.1e-5 at every threshold from 0.05 to 2.0 (15/15 same direction)",
      all_pos, "all 400 thresholds positive" if all_pos else "some threshold failed")
check("Results", "MLP escapes 15/15", escapes(mlp) == 15, f"{escapes(mlp)}/15")
check("Results", "token-only 15/15; positional-only 0/15; frozen random 0/15",
      (escapes(tok), escapes(pos), escapes(rand)) == (15, 0, 0), f"{escapes(tok)}, {escapes(pos)}, {escapes(rand)}")
check("Results", "positional-only and frozen random add the same 528,384 parameters",
      pos["n_params"] - post["n_params"] == rand["n_params"] - post["n_params"] == 528_384,
      f"{pos['n_params'] - post['n_params']:,} / {rand['n_params'] - post['n_params']:,}")
dmax = max(max(abs(x - y) for x, y in zip(r["final_loss_seeds"], post["final_loss_seeds"])) for r in (pos, rand))
check("Results", "failing variants track Post-LN within 0.001 loss per seed", dmax < 0.001, f"max {dmax:.5f}")

print("== Threats to Validity")
la = {k: load(f"layer_attn_n15/{k}")[k] for k in
      ("post_reinject", "post_reinject_detach", "post_reinject_mlp", "post_reinject_tok", "pre_ln")}
argmax = {k: [s.index(max(s)) + 1 for s in r["attn_prev_final_layers_seeds"]] for k, r in la.items()}
reinj_layers = [x for k in argmax if k != "pre_ln" for x in argmax[k]]
check("Threats", "in every seed the strongest attention on t-1 lies in blocks 12–16 with re-injection",
      min(reinj_layers) >= 12 and max(reinj_layers) <= 16, f"{min(reinj_layers)}–{max(reinj_layers)}")
check("Threats", "…but in block 1 with Pre-LN", set(argmax["pre_ln"]) == {1}, sorted(set(argmax["pre_ln"])))
same_seed = all(la[k]["final_loss_seeds"] == m[k]["final_loss_seeds"] for k in la)
check("Threats", "layer_attn_n15 rerun reproduces reinject_n15_paired seed for seed", same_seed, same_seed)

lr = load("threats_lr1e-4")
partial = lambda r: sum(v < math.log(64) - 0.1 for v in r["final_loss_seeds"])
check("Abs/Threats", "rescue persists at lr = 1e-4 (15/15)", escapes(lr["post_reinject"]) == 15,
      f"{escapes(lr['post_reinject'])}/15")
check("Threats", "at 1e-4 Post-LN progresses (all below chance − 0.1) but never escapes",
      escapes(lr["ln_post"]) == 0 and partial(lr["ln_post"]) == 15,
      f"escape {escapes(lr['ln_post'])}/15, progress {partial(lr['ln_post'])}/15")
mk = load("threats_modk")
dd = max(abs(x - y) for x, y in zip(mk["post_reinject"]["final_loss_seeds"], mk["ln_post"]["final_loss_seeds"]))
check("Abs/Threats", "not on modk: re-injection 0/15 and matches Post-LN seed for seed",
      escapes(mk["post_reinject"]) == 0 and partial(mk["post_reinject"]) == 0 and dd <= 0.00015,
      f"escape {escapes(mk['post_reinject'])}/15, max |diff| {dd:.5f}")
check("Threats", "Pre-LN progresses on 12/15 seeds on modk", partial(mk["pre_ln"]) == 12,
      f"{partial(mk['pre_ln'])}/15")

print()
print(f"{len(PASSES)} passed, {len(FAILS)} failed" + ("      ALL PASS" if not FAILS else f"      FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
