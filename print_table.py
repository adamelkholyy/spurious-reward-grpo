import argparse
import json
import os
import statistics

from tasks.numbers import row_range
from utils import BASE_MODELS, answer_counts, get_baseline_key

# always shown when present; other baselines only when a displayed run maps to them
DEFAULT_BASELINES = ("qwen", "llama", "olmo")

# orders q for the Hill-number plots of diversity tasks
HILL_QS = [i / 20 for i in range(0, 81)]  # 0 .. 4
# categorical hues, assigned in table order; a plot with more models is split into panels
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")

DEFAULT_SKIP_SUBSTRINGS = (
    "lr",
    "SCHED",
    "r2",
    "gt_high",
    "gt_low",
    "high_llam",
    "high_olm",
    "high_qwe",
)

_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _tex(s):
    return "".join(_LATEX_ESCAPES.get(c, c) for c in str(s))


def dataset_name(results_path):
    return os.path.basename(results_path).removeprefix("results_").removesuffix(".json")


def is_baseline(r):
    return r.get("is_baseline", r.get("label") in BASE_MODELS)


def load_rows(results_path, baseline_cache_path, skip_substrings):
    """Eval rows plus cache-only baselines, minus labels matching skip_substrings."""
    with open(results_path) as f:
        rows = [dict(r, _from_cache=False) for r in json.load(f)]

    # results file always takes precedence over the baseline cache
    seen = {r.get("eval_config") for r in rows if r.get("eval_config")}
    if baseline_cache_path and os.path.exists(baseline_cache_path):
        with open(baseline_cache_path) as f:
            for key, r in json.load(f).items():
                if key not in seen:
                    rows.append({**r, "eval_config": key, "is_baseline": True, "_from_cache": True})

    pats = tuple(s.lower() for s in skip_substrings)
    return [r for r in rows if not any(p in (r.get("label") or "").lower() for p in pats)]


def grouped_sections(rows, sort_by_model):
    """Yield ((n, temp, n_prob, max_tok, range), base_results, sections) for each n=256 eval config.

    ``range`` is the numbers task's ``"LO-HI"`` (None for other datasets), so each range gets its own table.
    """
    groups = {}
    for r in rows:
        cfg = (r.get("n_samples"), r.get("temperature"), r.get("n_problems"), r.get("max_tokens"), row_range(r))
        by_label = groups.setdefault(cfg, {})
        if r["_from_cache"] and r.get("label") in by_label:
            continue
        by_label[r.get("label")] = r  # last write wins

    for cfg, by_label in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if cfg[0] != 256:
            continue
        grp = list(by_label.values())
        base_results = {r["label"]: r for r in grp if is_baseline(r)}
        run_results = sorted((r for r in grp if not is_baseline(r)), key=lambda r: r.get("label") or "")
        shown = set(DEFAULT_BASELINES) | {get_baseline_key(r.get("label") or "") for r in run_results}
        baselines = [b for b in BASE_MODELS if b in shown and b in base_results]

        if sort_by_model:
            sections = []
            for b in [*BASE_MODELS, None]:  # None collects runs matching no base model
                sec = [base_results[b]] if b in baselines else []
                sec += [r for r in run_results if get_baseline_key(r.get("label") or "") == b]
                if sec:
                    sections.append(sec)
        else:
            sections = [[base_results[b] for b in baselines], run_results]
            sections = [s for s in sections if s]

        yield cfg, base_results, sections


def plot_hill_profiles(rows, title, path):
    """Hill number D_q against order q, one line per row (baselines dashed).

    Runs past the 8 colours spill into further panels with shared axes; every panel
    repeats the baselines, in the same colours, as the reference.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

    from tasks.capitals import hill_numbers

    bases = [r for r in rows if is_baseline(r)]
    runs = [r for r in rows if not is_baseline(r)]
    k = max(len(SERIES_COLORS) - len(bases), 1)
    base_colors = SERIES_COLORS[: len(bases)]
    run_colors = SERIES_COLORS[len(bases) :] or SERIES_COLORS[-1:]
    panels = [runs[i : i + k] for i in range(0, len(runs), k)] or [[]]
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(10, 4 * len(panels)), sharex=True, sharey=True, squeeze=False
    )
    for ax, panel in zip(axes[:, 0], panels):
        for r, color in [*zip(bases, base_colors), *zip(panel, run_colors)]:
            ys = hill_numbers(answer_counts(r["task_metrics"]), HILL_QS)
            ax.plot(HILL_QS, ys, color=color, linewidth=2, solid_capstyle="round",
                    linestyle="--" if is_baseline(r) else "-", label=r.get("label", "?"))
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel("Hill number $D_q$")
        ax.grid(color="#e1e0d9", linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    axes[-1, 0].set_xlabel("order q   (0: distinct, 1: exp Shannon, 2: inverse Simpson)")
    axes[0, 0].set_title(title, fontsize=10, loc="left")
    fig.tight_layout()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def print_ascii(sections, cells, head, caption):
    width = 55 + 10 + 12 * (len(head) - 2)
    print("\n" + "=" * width)
    print(caption)
    print("=" * width)
    print(f"{head[0]:<55}{head[1]:>10}" + "".join(f"{h:>12}" for h in head[2:]))
    print("-" * width)

    for i, sec in enumerate(sections):
        if i:
            print("-" * width)
        for r in sec:
            c = cells(r)
            label = c[0] if len(c[0]) <= 54 else c[0][:51] + "..."
            print(f"{label:<55}{c[1]:>10}" + "".join(f"{v:>12}" for v in c[2:]))
    print("=" * width)


def print_latex(sections, cells, head, caption):
    print()
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\begin{tabular}{l" + "r" * (len(head) - 1) + "}")
    print(r"\toprule")
    print(" & ".join(_tex(h) if not h.startswith("$") else h for h in head) + r" \\")
    print(r"\midrule")
    for i, sec in enumerate(sections):
        if i:
            print(r"\midrule")
        for r in sec:
            print(" & ".join(_tex(v) for v in cells(r)) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{" + _tex(caption.strip()) + "}")
    print(r"\end{table}")


def print_results_table(
    results_path="results/results_gsm8k.json",
    baseline_cache_path=None,
    latex=False,
    sort_by_model=False,
    skip_substrings=DEFAULT_SKIP_SUBSTRINGS,
    hill=True,
):
    """Reprint the eval script's results table from its output files.

    For diversity tasks (capitals, numbers) each table also gets a Hill-number plot (D_q against
    order q) at <results dir>/plots/<dataset>/hill_[<range>_]T<T>.png, unless hill=False.
    """
    rows = load_rows(results_path, baseline_cache_path, skip_substrings)
    diversity = any(answer_counts(r.get("task_metrics") or {}) is not None for r in rows)
    if diversity:
        from tasks.capitals import spread_metrics

    for (n, temp, n_prob, max_tok, rng), base_results, sections in grouped_sections(rows, sort_by_model):

        def cells(r):
            if diversity:
                # diversity columns replace the deltas (avg@n is the valid-answer rate here)
                counts = answer_counts(r["task_metrics"])
                spread = spread_metrics(counts)
                extra = [len(counts), f"{spread['effective_number']:.2f}", f"{spread['evenness']:.3f}"]
            elif is_baseline(r):
                extra = ["+0.0pp", "+0.0pp"]
            elif b := base_results.get(get_baseline_key(r.get("label") or "")):
                extra = [
                    f"{(r['accuracy'] - b['accuracy']) * 100:+.1f}pp",
                    f"{(r['pass_at_n'] - b['pass_at_n']) * 100:+.1f}pp",
                ]
            else:
                extra = ["--", "--"]
            return [
                r.get("label", "?"),
                f"{r['accuracy'] * 100:.1f}%",
                f"{r['pass_at_n'] * 100:.1f}%",
                *extra,
            ]

        caption = (
            f"{dataset_name(results_path)}{f' {rng}' if rng else ''} results   (n={n}, T={temp}, "
            f"{n_prob} problems, max_tokens={max_tok})"
        )
        if latex:
            extra = ["distinct", "$D_2$", "$E_2$"] if diversity else [r"$\Delta$avg", r"$\Delta$pass"]
            print_latex(sections, cells, ["model", f"avg@{n}", f"pass@{n}", *extra], caption)
        else:
            extra = ["distinct", "D2", "E2"] if diversity else ["Δavg", "Δpass"]
            print_ascii(sections, cells, ["model", "avg@n", f"pass@{n}", *extra], caption)

        if diversity and hill:
            name = dataset_name(results_path)
            path = os.path.join(
                os.path.dirname(results_path), "plots", name, f"hill_{rng + '_' if rng else ''}T{temp}.png"
            )
            plot_hill_profiles([r for sec in sections for r in sec], caption, path)
            print(f"Hill-number plot written to {path}")


def print_output_dist_table(results_path="results/results_output_dist.json"):
    with open(results_path) as f:
        results = json.load(f)

    print(f"{'model':<55}{'dataset':<50}{'mean entropy':>15}{'std':>15}")
    print("-" * 135)
    for r in results:
        h = r["distributions"]["mean_entropy"]
        print(f"{r['label']:<55}{r['dataset']:<50}{statistics.mean(h):>15.4f}{statistics.stdev(h):>15.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("results_path", nargs="?", default="results/results_gsm8k.json")
    p.add_argument("--baseline-cache", default=None)
    p.add_argument("--latex", action="store_true", help="emit a LaTeX table")
    p.add_argument(
        "--sort-by-model",
        action="store_true",
        help="group runs under their baseline family",
    )
    p.add_argument(
        "--no-skip",
        action="store_true",
        help="keep labels containing " + "/".join(DEFAULT_SKIP_SUBSTRINGS),
    )
    p.add_argument(
        "--no-hill",
        action="store_true",
        help="skip the Hill-number plots written for diversity tasks (capitals, numbers)",
    )
    p.add_argument(
        "--out_dist",
        action="store_true",
        help="print output distribution means and variances",
    )
    a = p.parse_args()

    if a.out_dist:
        print_output_dist_table()
    else:
        print_results_table(
            a.results_path,
            baseline_cache_path=a.baseline_cache,
            latex=a.latex,
            sort_by_model=a.sort_by_model,
            skip_substrings=() if a.no_skip else DEFAULT_SKIP_SUBSTRINGS,
            hill=not a.no_hill,
        )