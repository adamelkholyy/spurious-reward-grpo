"""Intermediate-value diversity reward for GSM8K.

Each correct sample is rewarded by how rare its solution strategy is within its
group, where a strategy is the set of intermediate values the solution computes
on its way to the answer. Wrong samples get 0, so diversity can never be bought
with wrong answers. For a group of G samples, sample j gets

    r_j = c_j * (alpha + (1 - alpha) / n_{s(j)})

with c_j the correctness, s(j) the strategy signature and n_s the number of
CORRECT samples in the group with signature s. Summed over a group this is
alpha * #correct + (1 - alpha) * #distinct correct strategies (as the numbers /
capitals inverse-frequency rewards, applied to strategies instead of answers).
alpha = 1 is the GT-only baseline.

Per sample: extract verified computations -> keep only those on the path to the
answer -> hash the remaining intermediate values -> count signatures in the group.
Correct samples with no checkable strategy ('direct' or 'invalid') get alpha and
no diversity credit.
"""

import ast
import math
import operator
import re
from collections import Counter
from dataclasses import dataclass

# ---------------------------------------------------------------- Step 1: extraction

NUM = r"-?\d[\d,]*(?:\.\d+)?"
ANNOT = re.compile(r"<<([^=<>]+)=([^<>]+)>>")
# Left-hand side of a free-text equation. Each number token is maximal (never followed
# by a digit, a comma or a decimal part), so a run splits into tokens one way only, and a
# match never starts inside a run: with NUM itself the regex backtracks exponentially on
# long digit runs ("1000000000000000...") and hangs the reward.
_TOK = r"\d[\d,]*(?:\.\d+)?(?![\d,]|\.\d)"
_OP = r"[()+\-*/×÷x^%$ ]"
EQN = re.compile(rf"(?<![\d()+\-*/×÷x^%$ ])((?:{_TOK}|{_OP})+?)=\s*\$?\s*({NUM})")

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


@dataclass(frozen=True)
class Step:
    inputs: tuple   # numeric literals on the left-hand side
    output: float   # verified result
    pos: int        # character offset, used to order steps


def to_num(s):
    return float(s.replace(",", "").replace("$", "").strip())


def safe_eval(expr):
    """Evaluate +-*/** arithmetic; return (value, literals used)."""
    lits = []

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            lits.append(float(n.value))
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 10:
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported")

    return ev(ast.parse(expr.strip(), mode="eval")), tuple(lits)


def clean(lhs):
    lhs = re.sub(r"^[\sx]+", "", lhs)                          # 'x' of a preceding word ("six 6*5")
    lhs = lhs.replace("×", "*").replace("÷", "/").replace("^", "**")
    lhs = re.sub(r"(?<=\d)\s*x\s*(?=[\d(])", "*", lhs)        # 3 x 4
    lhs = re.sub(r"(?<![\d,.])(\d[\d,.]*)\s*%", r"(\1/100)", lhs)  # 25% -> (25/100)
    return lhs.replace("$", "").replace(",", "")


def extract_steps(text, rel_tol=1e-4):
    steps = []
    matches = list(ANNOT.finditer(text)) or list(EQN.finditer(text))
    for m in matches:
        lhs = clean(m.group(1))
        if not re.search(r"[+\-*/]", lhs.strip().lstrip("-")):
            continue                      # "= 24" with no operation is not a step
        try:
            val, inputs = safe_eval(lhs)
            claimed = to_num(m.group(2))
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, RecursionError, MemoryError):
            continue
        if abs(val - claimed) > rel_tol * max(1.0, abs(claimed)):
            continue                      # arithmetic does not check out
        steps.append(Step(inputs, claimed, m.start()))
    return steps


# ---------------------------------------------------------------- Step 2: pruning

WORD_NUMS = {"half": 0.5, "twice": 2, "double": 2, "triple": 3, "thrice": 3, "dozen": 12,
             "pair": 2, "quarter": 0.25, "third": 3, "score": 20}
STANDARD_CONSTS = {1, 2, 3, 4, 7, 10, 12, 24, 30, 52, 60, 100, 365, 1000}


def key(v, sig=6):
    """Hashable numeric key: 6 significant figures, -0 folded to 0."""
    return float(f"{v:.{sig}g}") + 0.0


def problem_numbers(question):
    nums = {key(to_num(x)) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", question)}
    words = {key(v) for w, v in WORD_NUMS.items() if re.search(rf"\b{w}s?\b", question, re.I)}
    return nums | words


def prune(steps, answer, question):
    """Steps on the dependency path to `answer`, sorted by position.

    Returns [] when no verified step produces the answer, and None when a step on the
    path uses a constant that is neither given, standard, nor computed earlier.
    """
    givens = problem_numbers(question) | {key(c) for c in STANDARD_CONSTS}

    def producer(v, before):
        cands = [s for s in steps if key(s.output) == v and s.pos < before]
        return max(cands, key=lambda s: s.pos) if cands else None

    root = producer(key(answer), math.inf)
    if root is None:
        return []
    kept, stack = {root}, [root]
    while stack:
        s = stack.pop()
        for x in s.inputs:
            kx = key(x)
            p = producer(kx, s.pos)   # computed earlier beats "given": 24 may be 48/2, not hours/day
            if p is None:
                if kx in givens:
                    continue
                return None
            if p not in kept:
                kept.add(p)
                stack.append(p)
    return sorted(kept, key=lambda s: s.pos)


# ---------------------------------------------------------------- Step 3: signature

def signature(question, solution, answer):
    """('chain', frozenset of intermediate values) | ('direct',) | ('invalid',)."""
    try:
        answer = to_num(str(answer))
    except ValueError:
        return ("invalid",)
    chain = prune(extract_steps(solution), answer, question)
    if chain is None:
        return ("invalid",)
    if not chain:
        return ("direct",)
    return ("chain", frozenset(key(s.output) for s in chain) - {key(answer)})


# ---------------------------------------------------------------- Step 4: reward

def strategy_rewards(question, answer, completions, correct, alpha=0.5):
    """Rewards for one group (all completions for the same problem)."""
    sigs = [signature(question, c, answer) if ok else None for c, ok in zip(completions, correct)]
    counts = Counter(s for s in sigs if s is not None)

    def r(s):
        if s is None:
            return 0.0                    # wrong answer
        if s[0] != "chain":
            return alpha                  # correct, but no checkable strategy: no diversity credit
        return alpha + (1 - alpha) / counts[s]

    return [r(s) for s in sigs], sigs
