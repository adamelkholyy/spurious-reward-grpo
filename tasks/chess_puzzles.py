"""Chess puzzles task (Lichess puzzle database, first-move-only).

Data: a fixed 10k subset of ``Lichess/chess-puzzles`` (~5M rows, train split
only) stored at ``tasks/chess_puzzles/puzzles_10k.jsonl`` and built by
``python -m tasks.build_chess_subset``: puzzles with ``Rating <= 1500``,
``Popularity >= 80``, ``RatingDeviation < 100``, seeded uniform sample. Reading
the local file keeps training offline-safe (no 885 MB re-download per run).
500 rows are held out as the eval set (``holdout_n``), as for countdown.

Puzzle semantics: Lichess's ``FEN`` precedes the opponent's setup move
``Moves[0]``; the subset stores the position AFTER it (``position``) and the
solver's first move (``solution``, UCI). Only that first move is asked for.

Single prompt view (training == eval): FEN, side to move, and the legal moves
in SAN (sorted alphabetically so generation order isn't a hint); the model
returns one move inside ``<answer>...</answer>``. A completion is correct iff
the move in the LAST answer block is legal and either equals the solution move
or delivers checkmate (Lichess accepts any mating move as a puzzle solution).
SAN is the requested format, but legal UCI (``e2e4``) is accepted too.

The GRPO ``answer`` column carries ``{"position": fen, "solution": uci}`` as
one JSON string; the reward/grader decode it.

Rewards: ``chess`` (the ground-truth check above, default), ``chess_format``
(spurious baseline: any legal move in the answer block) and ``random``.
"""

from __future__ import annotations

import json
import os
import random
import re
from typing import Any, Dict, Optional, Tuple

import chess

from .base import DatasetSpec
from .registry import register_task

DATA_FILE = os.path.join(os.path.dirname(__file__), "chess_puzzles", "puzzles_10k.jsonl")

_CHESS_INSTRUCTION = (
    "Solve this chess puzzle.\n"
    "Position (FEN): {fen}\n"
    "{side} to move.\n"
    "Legal moves: {legal}\n"
    "Find the best move. Show your work in <think> </think> tags. And return "
    "your chosen move from the list of legal moves in <answer> </answer> tags, "
    "for example <answer> Nf3 </answer>."
)

# ---------------------------------------------------------------------------
# Extraction / grading
# ---------------------------------------------------------------------------
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
# Leading move numbers ("23." / "23...") and wrapper junk around the move.
_MOVE_NUM_RE = re.compile(r"^\d+\.(?:\.\.)?\s*")
_STRIP = " \t\n`$*\"'.,;:!?()[]{}"


def extract_move_text(text: str) -> Optional[str]:
    """Raw move string in the LAST ``<answer>`` block, or ``None``."""
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return None
    s = _MOVE_NUM_RE.sub("", matches[-1].strip()).strip(_STRIP)
    # "!"/"?" annotations sit after the move, e.g. "Qxf7#!!"; strip them.
    s = s.rstrip("!?")
    return s or None


def parse_move(board: chess.Board, s: str) -> Optional[chess.Move]:
    """Legal move from SAN (preferred) or UCI; ``None`` if neither parses."""
    try:
        return board.parse_san(s)
    except ValueError:
        pass
    try:
        mv = chess.Move.from_uci(s.lower())
    except ValueError:
        return None
    return mv if mv in board.legal_moves else None


def answered_move(completion: str, fen: str) -> Optional[chess.Move]:
    s = extract_move_text(completion)
    return None if s is None else parse_move(chess.Board(fen), s)


def grade(completion: str, fen: str, solution: str) -> bool:
    """Correct iff the answered move is the solution or gives checkmate."""
    mv = answered_move(completion, fen)
    if mv is None:
        return False
    if mv.uci() == solution:
        return True
    board = chess.Board(fen)
    board.push(mv)
    return board.is_checkmate()


def _decode_gold(gold: Any) -> Tuple[str, str]:
    d = json.loads(gold) if isinstance(gold, str) else gold
    return d["position"], d["solution"]


# ---------------------------------------------------------------------------
# Rewards (binary {0, 1}, matching the repo's other rewards)
# ---------------------------------------------------------------------------
def chess_reward(prompts, completions, answer, **kwargs):
    """1.0 iff the answered move solves the puzzle; else 0.0."""
    from utils import get_completion_text

    responses = [get_completion_text(c) for c in completions]
    return [1.0 if grade(r, *_decode_gold(a)) else 0.0
            for r, a in zip(responses, answer)]


def chess_format_reward(prompts, completions, answer, **kwargs):
    """Spurious/format-only baseline: 1.0 iff the answer block holds any legal
    move in the position — ignores whether it solves the puzzle."""
    from utils import get_completion_text

    responses = [get_completion_text(c) for c in completions]
    return [1.0 if answered_move(r, _decode_gold(a)[0]) is not None else 0.0
            for r, a in zip(responses, answer)]


def _register_reward():
    from rewards import REWARD_REGISTRY

    REWARD_REGISTRY.setdefault("chess", [chess_reward])
    REWARD_REGISTRY.setdefault("chess_format", [chess_format_reward])


_register_reward()


# ---------------------------------------------------------------------------
# Task spec
# ---------------------------------------------------------------------------
@register_task
class ChessPuzzlesTask(DatasetSpec):
    name = "chess"
    hf_path = "Lichess/chess-puzzles"   # source; rows read from DATA_FILE
    hf_config = None
    train_split = "train"
    eval_split = "train"   # train-only source...
    holdout_n = 500        # ...so hold out 500 puzzles for eval

    default_reward = "chess"
    allowed_rewards = ("chess", "chess_format", "random")

    system_prompt = None       # the user turn carries all instructions
    train_instruction = None

    def _load_rows(self):
        from datasets import load_dataset

        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(
                f"{DATA_FILE} missing; build it with "
                "`python -m tasks.build_chess_subset`."
            )
        return load_dataset("json", data_files=DATA_FILE, split="train")

    def _question(self, x: Dict[str, Any]) -> str:
        board = chess.Board(x["position"])
        legal = sorted(board.san(m) for m in board.legal_moves)
        return _CHESS_INSTRUCTION.format(
            fen=x["position"],
            side="White" if board.turn == chess.WHITE else "Black",
            legal=", ".join(legal),
        )

    def _gold(self, x: Dict[str, Any]) -> str:
        return json.dumps({"position": x["position"], "solution": x["solution"]})

    # ---- training ----
    def format_train_example(self, x, tokenizer):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": self._question(x)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": prompt, "answer": self._gold(x)}

    def build_train(self, tokenizer):
        ds, _ = self._split_holdout(self._load_rows())  # never train on eval rows
        return ds.map(
            lambda x: self.format_train_example(x, tokenizer),
            remove_columns=ds.column_names,
            load_from_cache_file=False,
        )

    # ---- eval ----
    def _load_eval_split(self, split=None):
        ds = self._load_rows()
        if self._holdout_active(split):
            _, ds = self._split_holdout(ds)
        return ds

    def load_eval(self, split=None, limit=None):
        ds = self._load_eval_split(split)
        if limit and limit > 0:
            ds = ds.select(range(min(limit, len(ds))))
        return [self._question(x) for x in ds], [self._gold(x) for x in ds]

    def sample_questions(self, split, num_prompts, seed):
        ds = self._load_eval_split(split)
        rng = random.Random(seed)
        idxs = rng.sample(range(len(ds)), k=min(num_prompts, len(ds)))
        return [self._question(ds[i]) for i in idxs]

    def eval_dataset_id(self, split=None) -> str:
        base = super().eval_dataset_id(split)
        return base.replace(self.hf_path, f"{self.hf_path}[subset10k]", 1)

    def extract_gold(self, answer_field) -> str:
        return str(answer_field)  # golds are pre-encoded JSON

    def extract_answer(self, completion: str) -> Optional[str]:
        return extract_move_text(completion)

    def make_grader(self):
        def _is_correct(completion: str, gold) -> bool:
            return grade(completion, *_decode_gold(gold))

        return _is_correct
