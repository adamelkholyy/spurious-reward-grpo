import argparse
import json
import os
import statistics

from utils import BASE_MODELS, get_baseline_key

# always shown when present; other baselines only when a displayed run maps to them
DEFAULT_BASELINES = ("qwen", "llama", "olmo")

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
    """Yield ((n, temp, n_prob, max_tok), base_results, sections) for each n=256 eval config."""
    groups = {}
    for r in rows:
        cfg = (r.get("n_samples"), r.get("temperature"), r.get("n_problems"), r.get("max_tokens"))
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
):
    """Reprint the eval script's results table from its output files."""
    rows = load_rows(results_path, baseline_cache_path, skip_substrings)

    for (n, temp, n_prob, max_tok), base_results, sections in grouped_sections(rows, sort_by_model):

        def cells(r):
            if is_baseline(r):
                d_avg = d_pass = "+0.0pp"
            elif b := base_results.get(get_baseline_key(r.get("label") or "")):
                d_avg = f"{(r['accuracy'] - b['accuracy']) * 100:+.1f}pp"
                d_pass = f"{(r['pass_at_n'] - b['pass_at_n']) * 100:+.1f}pp"
            elif "capitals" in results_path:
                d_avg, d_pass = r["task_metrics"]["distinct_countries"], "--"
            else:
                d_avg = d_pass = "--"
            return [
                r.get("label", "?"),
                f"{r['accuracy'] * 100:.1f}%",
                f"{r['pass_at_n'] * 100:.1f}%",
                d_avg,
                d_pass,
            ]

        caption = (
            f"{dataset_name(results_path)} results   (n={n}, T={temp}, {n_prob} problems, "
            f"max_tokens={max_tok})"
        )
        if latex:
            head = ["model", f"avg@{n}", f"pass@{n}", r"$\Delta$avg", r"$\Delta$pass"]
            print_latex(sections, cells, head, caption)
        else:
            head = ["model", "avg@n", f"pass@{n}", "Δavg", "Δpass"]
            print_ascii(sections, cells, head, caption)


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
        )
