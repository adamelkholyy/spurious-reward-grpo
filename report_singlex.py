"""Single-example experiment report: rebuilds EXPERIMENTS_singlex.md + bar charts from
results/, and (with --watch) emails a copy whenever a new single-example result lands.

    python report_singlex.py                 # rebuild once
    python report_singlex.py --email         # rebuild once and email
    python report_singlex.py --watch         # rebuild on change, email on new results (runs forever)
"""
import argparse
import glob
import json
import os
import smtplib
import time
from email.message import EmailMessage

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(ROOT, "EXPERIMENTS_singlex.md")
PLOTS = os.path.join(ROOT, "plots/singlex")
STATE = os.path.join(ROOT, "logs/singlex_report.state")
TO = "ae581@cam.ac.uk"

MODELS = ["qwen", "llama", "olmo"]
MODEL_NAMES = {"qwen": "Qwen2.5-1.5B", "llama": "Llama-3.2-1B", "olmo": "OLMo-2-1B"}
# eval dataset -> (train dataset prefix, title)
DATASETS = {
    "mbpp": ("mbpp", "MBPP"),
    "gsm8k": ("gsm8k", "GSM8K"),
    "countdown": ("countdown", "Countdown"),
    "aime2024": ("dapo", "AIME 2024 (trained on DAPO)"),
    "wordle": ("wordle", "Wordle"),
}
# reward -> (label template, colour, legend name, sweep log dir)
REWARDS = {
    "random": ("{p}_{m}_random_singlex-600", "#2a78d6", "Random", "singlex"),
    "gt": ("{p}_{m}_gt_singlex-600", "#eb6834", "Ground truth", "singlex_gt"),
    "format": ("{p}_{m}_format_singlex-600", "#1baf7a", "Format only", "singlex_format"),
}
OVERRIDES = {  # runs done under a different name with the same settings
    ("random", "gsm8k", "llama"): "gsm8k_llama_4800_random_singleex-600",
    ("format", "gsm8k", "llama"): "gsm8k_llama_1200_format_singlex-600",
}
BASE_COLOR = "#a3a29c"


def label(reward, ds, m):
    return OVERRIDES.get((reward, ds, m)) or REWARDS[reward][0].format(p=DATASETS[ds][0], m=m)


def load_results():
    out = {}
    for ds in DATASETS:
        try:
            out[ds] = json.load(open(os.path.join(ROOT, f"results/results_{ds}.json")))
        except (FileNotFoundError, json.JSONDecodeError):
            out[ds] = []
    return out


def base_acc(rows, m):
    c = [r for r in rows if r.get("is_baseline") and r["label"] == m and r.get("max_tokens") == 1024]
    c.sort(key=lambda r: r.get("n_samples", 0), reverse=True)
    return c[0]["accuracy"] if c else None


def table(results):
    """{ds: {m: {'base': acc, reward: acc}}}"""
    t = {}
    for ds, rows in results.items():
        by = {r["label"]: r["accuracy"] for r in rows}
        t[ds] = {m: {"base": base_acc(rows, m), **{rw: by.get(label(rw, ds, m)) for rw in REWARDS}} for m in MODELS}
    return t


def running():
    """run name currently in progress per sweep (last START with no DONE/FAIL after it)."""
    out = {}
    for rw, (_, _, _, d) in REWARDS.items():
        f = os.path.join(ROOT, "logs", d, "STATUS")
        if not os.path.exists(f):
            continue
        lines = open(f).read().splitlines()
        if lines and "sweep end" in lines[-1]:
            continue
        for ln in reversed(lines):
            parts = ln.split()
            if len(parts) > 3 and parts[2] in ("START", "DONE", "FAIL", "SKIP"):
                if parts[2] == "START":
                    out[rw] = (parts[3], parts[4])  # (train|eval, run)
                break
    return out


def pct(v):
    return "–" if v is None else (f"{v*100:.1f}" if v >= 0.01 or v == 0 else f"{v*100:.2f}")


def plot(t):
    os.makedirs(PLOTS, exist_ok=True)
    paths = []
    for ds, (_, title) in DATASETS.items():
        series = [("base", BASE_COLOR, "Base")] + [
            (rw, c, name) for rw, (_, c, name, _) in REWARDS.items() if any(t[ds][m][rw] is not None for m in MODELS)
        ]
        if len(series) == 1:
            continue
        k = len(series)
        w = 0.8 / k
        fig, ax = plt.subplots(figsize=(5.6, 3.1), dpi=160)
        fig.patch.set_facecolor("#fcfcfb")
        ax.set_facecolor("#fcfcfb")
        vals = []
        for i, (key, color, name) in enumerate(series):
            first = True
            for j, m in enumerate(MODELS):
                v = t[ds][m][key]
                x = j + (i - (k - 1) / 2) * w
                if v is None:
                    continue
                vals.append(v)
                ax.bar(x, v * 100, w - 0.02, color=color, label=name if first else None)
                first = False
                ax.text(x, v * 100, pct(v), ha="center", va="bottom", fontsize=6, color="#52514e")
        ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([MODEL_NAMES[m] for m in MODELS], fontsize=8)
        ax.set_xlim(-0.6, len(MODELS) - 0.4)
        ax.set_ylabel("Accuracy (avg@n, %)", fontsize=8, color="#52514e")
        ax.set_title(title, fontsize=9, loc="left", color="#0b0b0b")
        ax.tick_params(axis="y", labelsize=7, colors="#52514e")
        ax.tick_params(axis="x", length=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#d4d3cf")
        ax.yaxis.grid(True, color="#ebeae6", lw=0.6)
        ax.set_axisbelow(True)
        ymax = max(vals) * 100 if vals else 1
        ax.set_ylim(0, ymax * 1.45 if ymax > 0 else 1)
        fig.tight_layout()
        p = os.path.join(PLOTS, f"{ds}_base_vs_single.png")
        fig.savefig(p)
        plt.close(fig)
        paths.append((title, p))
    return paths


def write_md(t, run, plots):
    def cell(rw, ds, m):
        v = t[ds][m][rw]
        if v is not None:
            return pct(v)
        r = run.get(rw)
        if r and r[1] == label(rw, ds, m).rsplit("-", 1)[0]:
            return f"*{r[0]}ing*"
        return "todo"

    todo = sum(cell(rw, ds, m) == "todo" for rw in REWARDS for ds in DATASETS for m in MODELS)
    lines = [
        "# Single-example experiments",
        "",
        f"_Updated {time.strftime('%a %d %b %H:%M')}. Generated by `report_singlex.py`._",
        "",
        "Recipe: 1 training example, eps_low=1 / eps_high=0.1, constant LR. All scores are at **step 600**"
        " for every reward, so the comparison is like for like. Evals at n=256 (random) / n=32 (GT, format).",
        "",
        "## Status",
        "",
        f"Accuracy is avg@n in %. *training* / *evaluating* = in progress, todo = not started ({todo} left).",
        "",
        "| Dataset | Model | Base | Random | GT | Format |",
        "|---|---|---|---|---|---|",
    ]
    for ds, (_, title) in DATASETS.items():
        for m in MODELS:
            lines.append(
                f"| {title.split(' (')[0]} | {MODEL_NAMES[m]} | {pct(t[ds][m]['base'])} | "
                + " | ".join(cell(rw, ds, m) for rw in REWARDS)
                + " |"
            )
    lines += [
        "",
        "Sweeps: random = `archived_scripts/run_singlex.sh`; GT / format = `run_singlex_rewards.sh <gpu> gt|format`"
        " (progress in `logs/singlex_{gt,format}/STATUS`). The random-reward Wordle runs were"
        " lost to the 2 Nov eval hang and still need rerunning.",
        "",
        "## Single example vs base",
        "",
        "AIME has 30 problems, so differences below about 1% are noise.",
        "",
    ]
    for title, p in plots:
        lines.append(f"![{title}]({os.path.relpath(p, ROOT)})")
    open(MD, "w").write("\n".join(lines) + "\n")


def done_labels(t):
    return sorted(
        label(rw, ds, m) for ds in DATASETS for m in MODELS for rw in REWARDS if t[ds][m][rw] is not None
    )


def email(new, plots):
    msg = EmailMessage()
    msg["Subject"] = "Single-example results: " + (", ".join(new) if new else "current status")
    # Must be fully qualified: mail-serv.cl.cam.ac.uk rejects bare "user@flamingo" (504 5.5.2),
    # and the local exim accepts it anyway, so the failure was silent.
    msg["From"] = f"{os.environ.get('USER', 'ae581')}@cl.cam.ac.uk"
    msg["To"] = TO
    msg.set_content(open(MD).read())
    msg.add_attachment(open(MD, "rb").read(), maintype="text", subtype="markdown", filename=os.path.basename(MD))
    for _, p in plots:
        msg.add_attachment(open(p, "rb").read(), maintype="image", subtype="png", filename=os.path.basename(p))
    with smtplib.SMTP("localhost", 25, timeout=30) as s:
        s.send_message(msg)


def rebuild():
    t = table(load_results())
    plots = plot(t)
    write_md(t, running(), plots)
    return t, plots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", action="store_true")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=int, default=300)
    args = ap.parse_args()

    if not args.watch:
        _, plots = rebuild()
        if args.email:
            email([], plots)
        return

    seen = set(json.load(open(STATE))) if os.path.exists(STATE) else None
    last_sig = None
    while True:
        try:
            sig = [os.path.getmtime(f) for f in sorted(glob.glob(os.path.join(ROOT, "results/results_*.json")))]
            sig += [os.path.getmtime(f) for f in glob.glob(os.path.join(ROOT, "logs/singlex*/STATUS"))]
            if sig != last_sig:
                t, plots = rebuild()
                done = set(done_labels(t))
                if seen is None:
                    seen = done
                new = sorted(done - seen)
                if new:
                    email(new, plots)
                    print(time.strftime("%a %H:%M"), "emailed:", new, flush=True)
                seen = done
                json.dump(sorted(seen), open(STATE, "w"))
                last_sig = sig
        except Exception as e:  # keep watching through a half-written file etc.
            print(time.strftime("%a %H:%M"), "error:", repr(e), flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
