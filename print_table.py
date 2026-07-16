import json
import os

BASELINE_FAMILIES = ("qwen", "llama", "olmo")


def print_gsm8k_table(results_path="results_gsm8k.json",
                      baseline_cache_path=None):
    """Reprint the eval script's results table from its output files.

    Reads the persistent results file (a JSON list of result dicts) and,
    optionally, the baseline cache — only needed for baselines that were
    evaluated into the cache during a sweep whose --out was a DIFFERENT
    results file (anything in `results_path` already wins).

    Entries are grouped by eval config (n, temperature, n_problems,
    max_tokens) and one table is printed per group, since deltas are only
    meaningful within a config. Duplicate labels within a group keep the
    last-written entry.
    """
    with open(results_path) as f:
        rows = list(json.load(f))

    # Merge cache-only baselines (results file always takes precedence).
    seen = {r.get("eval_config") for r in rows if r.get("eval_config")}
    if baseline_cache_path and os.path.exists(baseline_cache_path):
        with open(baseline_cache_path) as f:
            for key, r in json.load(f).items():
                if key not in seen:
                    rows.append({**r, "eval_config": key,
                                 "is_baseline": True})

    def family(label):
        lbl = (label or "").lower()
        return next((f for f in BASELINE_FAMILIES if f in lbl), None)

    def is_baseline(r):
        # Entries from older script versions lack the flag; a baseline's
        # label is always exactly the family name.
        return r.get("is_baseline", r.get("label") in BASELINE_FAMILIES)

    # Group by everything score-affecting that results record.
    groups = {}
    for r in rows:
        cfg = (r.get("n_samples"), r.get("temperature"),
               r.get("n_problems"), r.get("max_tokens"))
        groups.setdefault(cfg, {})[r.get("label")] = r  # last write wins

    for (n, temp, n_prob, max_tok), by_label in sorted(
            groups.items(), key=lambda kv: str(kv[0])):
        grp = list(by_label.values())
        base_results = {r["label"]: r for r in grp if is_baseline(r)}
        run_results = [r for r in grp if not is_baseline(r)]
        k_cols = sorted({int(k) for r in grp
                         for k in r.get("pass_at_k_unbiased", {})})

        width = 55 + 10 + 12 + 12 * len(k_cols) + 12 + 12
        print("\n" + "=" * width)
        print(f"GSM8K results   (n={n}, T={temp}, {n_prob} problems, "
              f"max_tokens={max_tok})")
        print("=" * width)
        header = f"{'model':<55}{'avg@n':>10}{f'pass@{n}':>12}"
        for k in k_cols:
            header += f"{f'p@{k}(unb)':>12}"
        header += f"{'Δavg':>12}{'Δpass':>12}"
        print(header)
        print("-" * width)

        def fmt_row(r, delta_avg_str, delta_pass_str):
            label = r.get("label", "?")
            if len(label) > 54:
                label = label[:51] + "..."
            row = (f"{label:<55}"
                   f"{r['accuracy'] * 100:>9.1f}%"
                   f"{r['pass_at_n'] * 100:>11.1f}%")
            for k in k_cols:
                v = r.get("pass_at_k_unbiased", {}).get(str(k))
                row += f"{v * 100:>11.1f}%" if v is not None else f"{'--':>12}"
            return row + f"{delta_avg_str:>12}{delta_pass_str:>12}"

        for b in BASELINE_FAMILIES:
            if b in base_results:
                print(fmt_row(base_results[b], "+0.0pp", "+0.0pp"))
        if base_results and run_results:
            print("-" * width)

        for r in sorted(run_results, key=lambda r: r.get("label", "")):
            d_avg, d_pass = "--", "--"
            b = base_results.get(family(r.get("label")))
            if b:
                d_avg = f"{(r['accuracy'] - b['accuracy']) * 100:+.1f}pp"
                d_pass = f"{(r['pass_at_n'] - b['pass_at_n']) * 100:+.1f}pp"
            print(fmt_row(r, d_avg, d_pass))
        print("=" * width)


if __name__ == "__main__":
    # print_gsm8k_table("results_gsm8k.json")
    print_gsm8k_table("results_gsm8k.json", ".gsm8k_baseline_cache.json")