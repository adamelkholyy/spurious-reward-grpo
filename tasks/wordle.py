"""Wordle task (clembench / LM-Playschool dialogue game).

clembench's Wordle is a *multi-turn* game: the guesser (Player A) proposes a
5-letter word, a deterministic Game Master returns letter feedback
(green/yellow/red), and this repeats for up to six turns. See the clembench
paper (Chalamalasetti et al., 2023), Appendix D, for the exact prompt and
scoring; the interaction format reproduced here is:

    guess: apple
    explanation: <reasoning>
    ...
    guess_feedback:
    a<yellow> p<yellow> p<green> l<yellow> e<red>

with green = right letter & position, yellow = right letter / wrong position,
red = letter absent.

This GRPO harness is *single-turn* (prompt -> one completion -> scalar reward),
so we adapt Wordle into a "next-guess" task: each training example is a
flattened mid-game state (the guesser prompt plus a procedurally generated
history of prior guesses and their feedback), and the model must produce the
next `guess:`. Rewards (below) are faithful to clembench's own metrics and
specifically target the failure mode the paper documents — models "fail at
integrating the feedback across turns and using it to constrain their guesses".

IMPORTANT: the authoritative evaluation for the shared task is the *multi-turn*
clembench/playpen pipeline (https://github.com/lm-playpen/playpen). This task
produces training signal and a cheap single-turn proxy eval; it does not
replace the playpen episode evaluation.
"""

from __future__ import annotations

import os
import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

from .base import DatasetSpec
from .registry import register_task

# ---------------------------------------------------------------------------
# Prompt (clembench Wordle, traditional variant — paper Template D.1.1)
# ---------------------------------------------------------------------------
GUESSER_PROMPT = (
    "You are a language wizard who likes to guess words by using the given "
    "rules.\n\n"
    "Welcome to Wordle! You have six attempts to guess the target word, a "
    "valid English word of five lowercase letters (a-z). Please use the tags "
    '"guess:" and "explanation:" to provide a concise explanation for each '
    "guess.\n\n"
    'For instance, if your guess is "apple", your response should be\n'
    "guess: apple\n"
    "explanation: this is a common five-letter English word, and I am "
    "starting my guess with this word.\n\n"
    "After each guess, your answer will be validated, and you will receive "
    "feedback indicating which letters are correct (green), which letters are "
    "correct but in the wrong position (yellow), and which letters are "
    "incorrect (red). This feedback can be useful in determining which "
    "letters to include or exclude in your next guess.\n\n"
    'For example, the feedback for "apple" might be:\n'
    "guess_feedback:\n"
    "a<yellow> p<yellow> p<green> l<yellow> e<red>\n\n"
    "The explanation should contain details about how the guess_feedback is "
    "used to arrive at a new guess.\n\n"
    "Let's begin with your first guess."
)

WORD_LEN = 5
MAX_GUESSES = 6

# Sentinel marking the end of the fixed guesser instructions; the game
# transcript (real feedback) starts after this line.
_BEGIN_MARKER = "Let's begin with your first guess."

# ---------------------------------------------------------------------------
# Word lists (clembench uses 3blue1brown's lists: 2309 targets, 12953 guesses)
# ---------------------------------------------------------------------------
_WORDS_DIR = os.environ.get(
    "WORDLE_WORDS_DIR", os.path.join(os.path.dirname(__file__), "wordle_words")
)

# Minimal offline fallback so the task is runnable without the word-list files.
_FALLBACK_WORDS = (
    "apple grape lemon mango peach berry melon olive onion chili "
    "bread toast flour dough honey sugar candy syrup cream juice "
    "table chair couch shelf bench stool board plank glass metal "
    "river ocean beach cliff creek marsh swamp field plain ridge "
    "tiger zebra horse sheep goose robin eagle hawk crane finch "
    "brick stone slate amber pearl coral topaz jewel crown medal "
    "cloud storm rainy sunny windy foggy misty frost sleet vapor "
    "house cabin lodge villa manor tower crypt vault attic porch "
    "plant vine leaf stalk trunk bloom petal seeds roots thorn "
    "smart brave quick sharp swift steady eager keen bold noble "
    "light dark shade gleam flare spark blaze ember glint sheen "
    "north south early later above below inner outer front rear "
).split()
_FALLBACK_WORDS = sorted({w for w in _FALLBACK_WORDS if len(w) == WORD_LEN})


def _read_words(fname: str) -> List[str]:
    path = os.path.join(_WORDS_DIR, fname)
    try:
        with open(path) as f:
            return [w.strip().lower() for w in f if w.strip()]
    except OSError:
        return []


_TARGETS: Optional[List[str]] = None
_VALID: Optional[set] = None


def _load_word_lists() -> Tuple[List[str], set]:
    """Lazily load (targets, valid_guesses). Falls back to a small embedded
    list (with a warning) when the word-list files are absent."""
    global _TARGETS, _VALID
    if _TARGETS is not None:
        return _TARGETS, _VALID  # type: ignore[return-value]

    targets = _read_words("possible_words.txt")
    allowed = _read_words("allowed_words.txt")
    if not targets:
        import warnings

        warnings.warn(
            f"Wordle word lists not found in {_WORDS_DIR}; using a small "
            f"built-in fallback ({len(_FALLBACK_WORDS)} words). For faithful "
            "clembench behaviour place possible_words.txt / allowed_words.txt "
            "there (3blue1brown lists) or set $WORDLE_WORDS_DIR.",
            stacklevel=2,
        )
        targets = list(_FALLBACK_WORDS)
    valid = set(targets) | set(allowed)
    _TARGETS, _VALID = targets, valid
    return targets, valid


# ---------------------------------------------------------------------------
# Wordle mechanics
# ---------------------------------------------------------------------------
def compute_feedback(guess: str, target: str) -> List[str]:
    """Return per-position ['green'|'yellow'|'red'] for guess vs target.

    Standard Wordle duplicate handling: greens first, then yellows drawn from
    the remaining (non-green) target letter counts.
    """
    guess, target = guess.lower(), target.lower()
    res = ["red"] * WORD_LEN
    counts: Dict[str, int] = {}
    for ch in target:
        counts[ch] = counts.get(ch, 0) + 1
    for i in range(WORD_LEN):
        if guess[i] == target[i]:
            res[i] = "green"
            counts[guess[i]] -= 1
    for i in range(WORD_LEN):
        if res[i] == "green":
            continue
        ch = guess[i]
        if counts.get(ch, 0) > 0:
            res[i] = "yellow"
            counts[ch] -= 1
    return res


def render_feedback(guess: str, colors: Sequence[str]) -> str:
    """'a<yellow> p<green> ...' — the clembench guess_feedback token string."""
    return " ".join(f"{ch}<{c}>" for ch, c in zip(guess.lower(), colors))


def closeness(colors: Sequence[str]) -> int:
    """clembench Closeness for one guess: +5 per green, +3 per yellow (0..25)."""
    return sum(5 if c == "green" else 3 if c == "yellow" else 0 for c in colors)


# --- parsing ---------------------------------------------------------------
_GUESS_RE = re.compile(r"guess:\s*([a-zA-Z]+)", re.IGNORECASE)
_FB_LINE_RE = re.compile(
    r"guess_feedback:\s*((?:[a-zA-Z]<(?:green|yellow|red)>\s*){%d})" % WORD_LEN,
    re.IGNORECASE,
)
_FB_TOK_RE = re.compile(r"([a-zA-Z])<(green|yellow|red)>", re.IGNORECASE)


def parse_guess(text: str) -> Optional[str]:
    """Extract the model's guessed word (first 'guess:' occurrence)."""
    m = _GUESS_RE.search(text)
    if not m:
        return None
    return m.group(1).lower()


def parse_feedback_history(prompt_text: str) -> List[List[Tuple[str, str]]]:
    """Reconstruct prior turns from the rendered guess_feedback lines in a
    prompt. Each turn -> [(letter, color) x5].

    Only scans the transcript *after* the guesser prompt's "Let's begin..."
    marker, so the illustrative guess_feedback example in the boilerplate is
    not mistaken for real game history.
    """
    idx = prompt_text.rfind(_BEGIN_MARKER)
    scan = prompt_text[idx + len(_BEGIN_MARKER):] if idx != -1 else prompt_text
    rows = []
    for m in _FB_LINE_RE.finditer(scan):
        toks = [(l.lower(), c.lower()) for l, c in _FB_TOK_RE.findall(m.group(1))]
        if len(toks) == WORD_LEN:
            rows.append(toks)
    return rows


# --- constraint model (for feedback-consistency) ---------------------------
def _constraints(rows: List[List[Tuple[str, str]]]):
    """Aggregate hard-mode constraints from prior feedback rows.

    Returns (green_pos, banned_pos, min_count, max_count):
      green_pos[i]=letter (position must be that letter)
      banned_pos: set of (i, letter) (letter must NOT be at i, but must appear)
      min_count[letter], max_count[letter]: letter-multiplicity bounds
    """
    green_pos: Dict[int, str] = {}
    banned_pos = set()
    min_count: Dict[str, int] = {}
    max_count: Dict[str, int] = {}
    for row in rows:
        gy: Dict[str, int] = {}
        red: Dict[str, int] = {}
        for i, (ch, color) in enumerate(row):
            if color == "green":
                green_pos[i] = ch
                gy[ch] = gy.get(ch, 0) + 1
            elif color == "yellow":
                banned_pos.add((i, ch))
                gy[ch] = gy.get(ch, 0) + 1
            else:
                red[ch] = red.get(ch, 0) + 1
        for ch, n in gy.items():
            min_count[ch] = max(min_count.get(ch, 0), n)
        for ch in red:
            # a red for ch means the target has exactly gy[ch] copies of it
            cap = gy.get(ch, 0)
            max_count[ch] = min(max_count.get(ch, WORD_LEN), cap)
    return green_pos, banned_pos, min_count, max_count


def is_consistent(guess: str, rows: List[List[Tuple[str, str]]]) -> bool:
    """True if `guess` respects every constraint implied by prior feedback
    (i.e. it is a legal hard-mode guess given the history)."""
    if len(guess) != WORD_LEN:
        return False
    green_pos, banned_pos, min_count, max_count = _constraints(rows)
    for i, ch in green_pos.items():
        if guess[i] != ch:
            return False
    for i, ch in banned_pos:
        if guess[i] == ch or ch not in guess:
            return False
    for ch, m in min_count.items():
        if guess.count(ch) < m:
            return False
    for ch, m in max_count.items():
        if guess.count(ch) > m:
            return False
    return True


# ---------------------------------------------------------------------------
# Game-state rendering (prompt body shared by training and eval)
# ---------------------------------------------------------------------------
def render_user_turn(history: List[Tuple[str, List[str]]]) -> str:
    """Full guesser user-turn text: the base prompt plus a transcript of prior
    (guess, feedback) turns. `history` is a list of (guess, colors)."""
    parts = [GUESSER_PROMPT]
    for guess, colors in history:
        parts.append(f"guess: {guess}")
        parts.append("guess_feedback:\n" + render_feedback(guess, colors))
    return "\n\n".join(parts)


def _sample_history(target, k, valid_list, rng, consistent_chain):
    """Build k prior (guess, colors) turns for a game whose answer is target."""
    history: List[Tuple[str, List[str]]] = []
    used = set()
    rows: List[List[Tuple[str, str]]] = []
    for _ in range(k):
        pool = valid_list
        if consistent_chain:
            # bias toward guesses a real player could make given feedback so far
            cand = [w for w in valid_list if w != target and w not in used
                    and is_consistent(w, rows)]
            pool = cand or valid_list
        for _try in range(8):
            g = rng.choice(pool)
            if g != target and g not in used:
                break
        used.add(g)
        colors = compute_feedback(g, target)
        history.append((g, colors))
        rows.append(list(zip(g, colors)))
    return history


# ---------------------------------------------------------------------------
# Reward functions — registered into rewards.REWARD_REGISTRY at import time.
#
# All are format-gated: a completion that doesn't yield a 5-letter `guess:`
# scores 0 (mirrors clembench aborting an unparsable move). Names mirror the
# GSM8K reward family: a shaped default plus true/metric/spurious variants.
# ---------------------------------------------------------------------------
def _per_sample(prompt: str, completion_text: str, target: str) -> dict:
    """Compute all wordle signals for one (prompt, completion, target)."""
    _, valid = _load_word_lists()
    target = str(target).lower()
    guess = parse_guess(completion_text)
    rows = parse_feedback_history(prompt)

    ok_format = bool(guess) and len(guess) == WORD_LEN and guess.isalpha()
    valid_word = ok_format and guess in valid
    correct = ok_format and guess == target
    if ok_format:
        colors = compute_feedback(guess, target)
        close = closeness(colors) / (5 * WORD_LEN)  # -> [0, 1]
        consistent = valid_word and is_consistent(guess, rows)
    else:
        close, consistent = 0.0, False
    return {
        "format": ok_format,
        "valid_word": valid_word,
        "correct": correct,
        "closeness": close,
        "consistent": consistent,
    }


def _texts(completions):
    from utils import get_completion_text

    return [get_completion_text(c) for c in completions]


def _log(metrics: dict):
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics, commit=False)
    except Exception:
        pass


def wordle_reward(prompts, completions, answer, **kwargs):
    """Shaped default. 0 if unparsable/invalid word; 1.0 if exactly correct;
    otherwise 0.5*closeness + 0.5*consistency. The consistency term is the key
    signal: it rewards guesses that actually respect the letter feedback."""
    texts = _texts(completions)
    stats = [_per_sample(p, t, a) for p, t, a in zip(prompts, texts, answer)]
    scores = []
    for s in stats:
        if not s["valid_word"]:
            scores.append(0.0)
        elif s["correct"]:
            scores.append(1.0)
        else:
            scores.append(0.5 * s["closeness"] + 0.5 * (1.0 if s["consistent"] else 0.0))
    n = max(len(stats), 1)
    _log({
        "train/wordle_success": sum(s["correct"] for s in stats) / n,
        "train/wordle_valid_word": sum(s["valid_word"] for s in stats) / n,
        "train/wordle_consistent": sum(s["consistent"] for s in stats) / n,
        "train/wordle_closeness": sum(s["closeness"] for s in stats) / n,
        "train/wordle_format": sum(s["format"] for s in stats) / n,
    })
    return scores


def wordle_success_reward(prompts, completions, answer, **kwargs):
    """True reward: 1.0 iff the guess equals the target (clembench Success)."""
    texts = _texts(completions)
    return [1.0 if _per_sample(p, t, a)["correct"] else 0.0
            for p, t, a in zip(prompts, texts, answer)]


def wordle_closeness_reward(prompts, completions, answer, **kwargs):
    """Dense reward = clembench Closeness normalised to [0,1] (+5 green/+3
    yellow per position, /25). Format-gated."""
    texts = _texts(completions)
    return [_per_sample(p, t, a)["closeness"]
            for p, t, a in zip(prompts, texts, answer)]


def wordle_consistent_reward(prompts, completions, answer, **kwargs):
    """1.0 iff the guess is a valid word AND respects all prior feedback
    (a legal hard-mode move). Directly targets feedback integration."""
    texts = _texts(completions)
    return [1.0 if _per_sample(p, t, a)["consistent"] else 0.0
            for p, t, a in zip(prompts, texts, answer)]


def wordle_format_reward(completions, **kwargs):
    """Spurious/format-only baseline (cf. GSM8K box_only): 1.0 iff the response
    contains a well-formed 5-letter `guess:` — ignores correctness entirely."""
    texts = _texts(completions)
    out = []
    for t in texts:
        g = parse_guess(t)
        out.append(1.0 if (g and len(g) == WORD_LEN and g.isalpha()) else 0.0)
    return out


def _register_rewards():
    """Add the wordle rewards to the shared registry (so --reward <name> and
    the trainer's dynamic CLI choices pick them up). Kept here so the whole
    task — prompt, data, grading, and rewards — lives in one file."""
    from rewards import REWARD_REGISTRY

    REWARD_REGISTRY.update({
        "wordle": [wordle_reward],
        "wordle_success": [wordle_success_reward],
        "wordle_closeness": [wordle_closeness_reward],
        "wordle_consistent": [wordle_consistent_reward],
        "wordle_format": [wordle_format_reward],
    })


_register_rewards()


# ---------------------------------------------------------------------------
# Task spec
# ---------------------------------------------------------------------------
@register_task
class WordleTask(DatasetSpec):
    name = "wordle"
    hf_path = "clembench/wordle"   # informational; data is generated locally
    hf_config = None

    default_reward = "wordle"
    allowed_rewards = (
        "wordle", "wordle_success", "wordle_closeness",
        "wordle_consistent", "wordle_format",
    )

    # clembench puts all instructions in the (user) game-master turn; no system
    # prompt. The "question" for each example IS the full rendered user turn.
    system_prompt = None
    train_instruction = None

    # Data-generation knobs (env-overridable so runs stay reproducible).
    n_instances = int(os.environ.get("WORDLE_N_INSTANCES", "2000"))
    n_eval = int(os.environ.get("WORDLE_N_EVAL", "500"))
    max_prior_guesses = int(os.environ.get("WORDLE_MAX_PRIOR", "4"))
    consistent_history = os.environ.get("WORDLE_CONSISTENT_HISTORY", "0") == "1"
    seed = int(os.environ.get("WORDLE_SEED", "0"))

    # ---- shared state generation ----
    def _make_examples(self, n: int, seed: int) -> List[Dict[str, str]]:
        targets, valid = _load_word_lists()
        valid_list = sorted(valid)
        rng = random.Random(seed)
        out = []
        for _ in range(n):
            target = rng.choice(targets)
            k = rng.randint(0, self.max_prior_guesses)
            history = _sample_history(
                target, k, valid_list, rng, self.consistent_history
            )
            out.append({"user_turn": render_user_turn(history), "answer": target})
        return out

    # ---- training ----
    def format_train_example(self, x, tokenizer):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": x["user_turn"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": prompt, "answer": x["answer"]}

    def build_train(self, tokenizer):
        from datasets import Dataset

        rows = self._make_examples(self.n_instances, self.seed)
        ds = Dataset.from_list(rows)
        return ds.map(
            lambda x: self.format_train_example(x, tokenizer),
            remove_columns=ds.column_names,
            load_from_cache_file=False,
        )

    # ---- eval (single-turn "next guess" proxy; playpen is authoritative) ----
    def load_eval(self, split=None, limit=None):
        rows = self._make_examples(self.n_eval, self.seed + 1)
        if limit and limit > 0:
            rows = rows[:limit]
        questions = [r["user_turn"] for r in rows]
        golds = [r["answer"] for r in rows]
        return questions, golds

    def sample_questions(self, split, num_prompts, seed):
        rows = self._make_examples(num_prompts, seed)
        return [r["user_turn"] for r in rows]

    def extract_gold(self, answer_field) -> str:
        return str(answer_field).strip().lower()

    def extract_answer(self, completion: str) -> Optional[str]:
        return parse_guess(completion)

    def make_grader(self):
        def _is_correct(completion: str, gold) -> bool:
            g = parse_guess(completion)
            return g is not None and g == str(gold).strip().lower()

        return _is_correct

    def eval_dataset_id(self, split=None) -> str:
        return f"clembench:wordle:next_guess:{split or 'gen'}"
