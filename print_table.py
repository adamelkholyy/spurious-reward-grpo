import argparse
import json
import os
import numpy as np

BASELINE_FAMILIES = ("qwen", "llama", "olmo")


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


def print_gsm8k_table(
    results_path="results/results_gsm8k.json",
    baseline_cache_path=None,
    latex=False,
    sort_by_model=False,
    skip_substrings=DEFAULT_SKIP_SUBSTRINGS,
):
    """Reprint the eval script's results table from its output files."""
    with open(results_path) as f:
        rows = [dict(r, _from_cache=False) for r in json.load(f)]

    # Merge cache-only baselines (results file always takes precedence).
    seen = {r.get("eval_config") for r in rows if r.get("eval_config")}
    if baseline_cache_path and os.path.exists(baseline_cache_path):
        with open(baseline_cache_path) as f:
            for key, r in json.load(f).items():
                if key not in seen:
                    rows.append(
                        {
                            **r,
                            "eval_config": key,
                            "is_baseline": True,
                            "_from_cache": True,
                        }
                    )

    if skip_substrings:
        pats = tuple(s.lower() for s in skip_substrings)
        rows = [
            r
            for r in rows
            if not any(p in (r.get("label") or "").lower() for p in pats)
        ]

    def family(label):
        lbl = (label or "").lower()
        return next((f for f in BASELINE_FAMILIES if f in lbl), None)

    def is_baseline(r):
        # label is always exactly the family name.
        return r.get("is_baseline", r.get("label") in BASELINE_FAMILIES)

    groups = {}
    for r in rows:
        cfg = (
            r.get("n_samples"),
            r.get("temperature"),
            r.get("n_problems"),
            r.get("max_tokens"),
        )
        by_label = groups.setdefault(cfg, {})

        if r["_from_cache"] and r.get("label") in by_label:
            continue
        by_label[r.get("label")] = r  # last write wins

    for (n, temp, n_prob, max_tok), by_label in sorted(
        groups.items(), key=lambda kv: str(kv[0])
    ):
        grp = list(by_label.values())
        base_results = {r["label"]: r for r in grp if is_baseline(r)}
        run_results = [r for r in grp if not is_baseline(r)]
        k_cols = sorted({int(k) for r in grp for k in r.get("pass_at_k_unbiased", {})})

        by_label_sort = lambda r: (r.get("label") or "")

        if sort_by_model:
            sections = []
            for b in BASELINE_FAMILIES:
                sec = [base_results[b]] if b in base_results else []
                sec += sorted(
                    (r for r in run_results if family(r.get("label")) == b),
                    key=by_label_sort,
                )
                if sec:
                    sections.append(sec)
            orphans = sorted(
                (r for r in run_results if family(r.get("label")) is None),
                key=by_label_sort,
            )
            if orphans:
                sections.append(orphans)
        else:
            sections = [
                [base_results[b] for b in BASELINE_FAMILIES if b in base_results],
                sorted(run_results, key=by_label_sort),
            ]
            sections = [s for s in sections if s]

        def deltas(r):
            if is_baseline(r):
                return "+0.0pp", "+0.0pp"
            b = base_results.get(family(r.get("label")))
            if not b:
                return "--", "--"
            return (
                f"{(r['accuracy'] - b['accuracy']) * 100:+.1f}pp",
                f"{(r['pass_at_n'] - b['pass_at_n']) * 100:+.1f}pp",
            )

        def cells(r):
            d_avg, d_pass = deltas(r)
            out = [
                r.get("label", "?"),
                f"{r['accuracy'] * 100:.1f}%",
                f"{r['pass_at_n'] * 100:.1f}%",
            ]
            for k in k_cols:
                v = r.get("pass_at_k_unbiased", {}).get(str(k))
                out.append(f"{v * 100:.1f}%" if v is not None else "--")
            return out + [d_avg, d_pass]

        caption = (
            f"GSM8K results   (n={n}, T={temp}, {n_prob} problems, "
            f"max_tokens={max_tok})"
        )

        if latex:
            _print_latex(sections, cells, k_cols, n, caption)
        else:
            _print_ascii(sections, cells, k_cols, n, caption)


def _print_ascii(sections, cells, k_cols, n, caption):
    width = 55 + 10 + 12 + 12 * len(k_cols) + 12 + 12
    print("\n" + "=" * width)
    print(caption)
    print("=" * width)
    header = f"{'model':<55}{'avg@n':>10}{f'pass@{n}':>12}"
    for k in k_cols:
        header += f"{f'p@{k}(unb)':>12}"
    header += f"{'Δavg':>12}{'Δpass':>12}"
    print(header)
    print("-" * width)

    for i, sec in enumerate(sections):
        if i:
            print("-" * width)
        for r in sec:
            c = cells(r)
            label = c[0] if len(c[0]) <= 54 else c[0][:51] + "..."
            row = f"{label:<55}{c[1]:>10}"
            for v in c[2:]:
                row += f"{v:>12}"
            print(row)
    print("=" * width)


def _print_latex(sections, cells, k_cols, n, caption):
    ncols = 5 + len(k_cols)
    head = ["model", f"avg@{n}", f"pass@{n}"]
    head += [f"p@{k} (unb.)" for k in k_cols]
    head += [r"$\Delta$avg", r"$\Delta$pass"]

    print()
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\begin{tabular}{l" + "r" * (ncols - 1) + "}")
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


def load_results(path="results_output_dist.json"):
    """Load the results file and return a {label: entry} dict."""
    with open(path) as f:
        results = json.load(f)
    return {e["label"]: e for e in results}


def print_output_dist_table(results_path="results/results_output_dist.json"):
    with open(results_path) as f:
        results = json.load(f)

    print(f"{'model':<65}{'mean entropy':>15}{'std':>15}")
    print("-" * 80)

    print(len(results))

    for r in results:
        e = r["label"]
        try:
            mean = r["stats"]["mean_entropy"]["mean"]
            vari = r["stats"]["mean_entropy"]["std"]
            print(f"{e:<65}{mean:>15.4f}{vari:>15.4f}")

        except:
            pass


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
        print_gsm8k_table(
            a.results_path,
            baseline_cache_path=a.baseline_cache,
            latex=a.latex,
            sort_by_model=a.sort_by_model,
            skip_substrings=() if a.no_skip else DEFAULT_SKIP_SUBSTRINGS,
        )

    # print_output_dist_table()
