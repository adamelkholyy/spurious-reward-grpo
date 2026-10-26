import argparse
import json
import statistics

from print_table import (
    DEFAULT_SKIP_SUBSTRINGS,
    dataset_name,
    grouped_sections,
    is_baseline,
    load_rows,
    print_ascii,
    print_latex,
    print_output_dist_table,
)
from utils import get_baseline_key


def _hf_path(dataset_id):
    """'openai/gsm8k:main:test' -> 'openai/gsm8k'"""
    return dataset_id.split(":")[0]


def load_entropy_stats(path):
    """{(hf dataset path, label): (mean, std) of per-rollout mean token entropy}.

    Keyed by dataset too: each dataset has its own baseline entries under the same label.
    """
    with open(path) as f:
        results = json.load(f)
    out = {}
    for r in results:
        h = r["distributions"]["mean_entropy"]
        out[(_hf_path(r["dataset"]), r["label"])] = (statistics.mean(h), statistics.stdev(h))
    return out


def _pct_change(new, base):
    """Relative change in percent: (new - base) / base * 100."""
    if base == 0:
        return "--"
    return f"{(new - base) / base * 100:+.1f}%"


def print_results_table(
    results_path="results/results_gsm8k.json",
    baseline_cache_path=None,
    dist_path="results/results_output_dist.json",
    latex=False,
    sort_by_model=False,
    skip_substrings=DEFAULT_SKIP_SUBSTRINGS,
):
    """Reprint the eval script's results table from its output files.

    Differences vs. the pp version (print_table.py):
      * avg / pass deltas are relative (% increase over the base model).
      * two extra columns give the raw difference (variant - base) in the mean
        and std of the output-entropy distribution.
    """
    rows = load_rows(results_path, baseline_cache_path, skip_substrings)
    entropy = load_entropy_stats(dist_path)

    def entropy_of(r, label):
        return entropy.get((_hf_path(json.loads(r["eval_config"])["dataset"]), label))

    for (n, temp, n_prob, max_tok), base_results, sections in grouped_sections(rows, sort_by_model):

        def cells(r):
            mine = entropy_of(r, r.get("label"))
            if is_baseline(r):
                # baseline rows show absolute entropy rather than a delta
                d_avg = d_pass = "+0.0%"
                d_mean, d_std = (f"{mine[0]:.4f}", f"{mine[1]:.4f}") if mine else ("--", "--")
            else:
                fam = get_baseline_key(r.get("label") or "")
                b = base_results.get(fam)
                if b:
                    d_avg = _pct_change(r["accuracy"], b["accuracy"])
                    d_pass = _pct_change(r["pass_at_n"], b["pass_at_n"])
                else:
                    d_avg = d_pass = "--"
                base = entropy_of(b, fam) if b else None
                if mine and base:
                    d_mean, d_std = f"{mine[0] - base[0]:+.4f}", f"{mine[1] - base[1]:+.4f}"
                else:
                    d_mean = d_std = "--"
            return [
                r.get("label", "?"),
                f"{r['accuracy'] * 100:.1f}%",
                f"{r['pass_at_n'] * 100:.1f}%",
                d_avg,
                d_pass,
                d_mean,
                d_std,
            ]

        caption = (
            f"{dataset_name(results_path)} results   (n={n}, T={temp}, {n_prob} problems, "
            f"max_tokens={max_tok}; accuracy deltas are relative % change, "
            f"entropy deltas are raw differences vs. base)"
        )
        if latex:
            head = [
                "model",
                f"avg@{n}",
                f"pass@{n}",
                r"$\Delta$avg (\%)",
                r"$\Delta$pass (\%)",
                r"$\Delta H_{\mathrm{mean}}$",
                r"$\Delta H_{\mathrm{std}}$",
            ]
            print_latex(sections, cells, head, caption)
        else:
            head = ["model", "avg@n", f"pass@{n}", "Δavg %", "Δpass %", "ΔH mean", "ΔH std"]
            print_ascii(sections, cells, head, caption)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("results_path", nargs="?", default="results/results_gsm8k.json")
    p.add_argument("--baseline-cache", default=None)
    p.add_argument(
        "--dist-path",
        default="results/results_output_dist.json",
        help="output-distribution results used for the entropy delta columns",
    )
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
        print_output_dist_table(a.dist_path)
    else:
        print_results_table(
            a.results_path,
            baseline_cache_path=a.baseline_cache,
            dist_path=a.dist_path,
            latex=a.latex,
            sort_by_model=a.sort_by_model,
            skip_substrings=() if a.no_skip else DEFAULT_SKIP_SUBSTRINGS,
        )
