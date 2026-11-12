"""Build the local chess-puzzle subset used by ``tasks/chess_puzzles.py``.

Run once (needs internet; ~885 MB of parquet goes to the HF cache)::

    python -m tasks.build_chess_subset   # -> tasks/chess_puzzles/puzzles_10k.jsonl

(Run as a module: running the file directly puts tasks/ on sys.path and
tasks/numbers.py then shadows the stdlib ``numbers`` module.)

Source: ``Lichess/chess-puzzles`` (~5M rows, train split only). Filter:
``Rating <= 1500``, ``Popularity >= 80``, ``RatingDeviation < 100`` (easy,
well-liked, settled-rating puzzles; drops noisy/broken ones), then a seeded
uniform sample of ``N`` rows, sorted by ``PuzzleId`` so the file is stable.

Lichess semantics: ``FEN`` is the position BEFORE the opponent's move and
``Moves[0]`` is that move; the puzzle starts after it. We precompute that
position (``position``) and the solver's first move (``solution`` in UCI,
``solution_san``) so the task file never has to replay it.
"""

from __future__ import annotations

import argparse
import json
import os

import chess
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "Lichess/chess-puzzles"
FILES = [f"data/train-0000{i}-of-00003.parquet" for i in range(3)]
COLS = ["PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation",
        "Popularity", "NbPlays", "Themes"]
OUT = os.path.join(os.path.dirname(__file__), "chess_puzzles", "puzzles_10k.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_rating", type=int, default=1500)
    ap.add_argument("--min_popularity", type=int, default=80)
    ap.add_argument("--max_rd", type=int, default=100)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    tables = []
    for f in FILES:
        path = hf_hub_download(REPO, f, repo_type="dataset")
        t = pq.read_table(path, columns=COLS)
        mask = pc.and_(
            pc.and_(pc.less_equal(t["Rating"], args.max_rating),
                    pc.greater_equal(t["Popularity"], args.min_popularity)),
            pc.less(t["RatingDeviation"], args.max_rd),
        )
        tables.append(t.filter(mask))
        print(f"{f}: {len(tables[-1])}/{len(t)} pass filter")
    rows = [r for t in tables for r in t.to_pylist()]
    rows.sort(key=lambda r: r["PuzzleId"])  # order-independent of file layout
    print(f"{len(rows)} candidates")

    rng = np.random.default_rng(args.seed)
    picked = sorted(rng.choice(len(rows), size=min(args.n, len(rows)), replace=False))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    kept = 0
    with open(args.out, "w") as fh:
        for i in picked:
            r = rows[int(i)]
            moves = r["Moves"].split()
            board = chess.Board(r["FEN"])
            board.push_uci(moves[0])          # opponent's setup move
            sol = chess.Move.from_uci(moves[1])
            if sol not in board.legal_moves:  # defensive; shouldn't happen
                continue
            fh.write(json.dumps({
                "puzzle_id": r["PuzzleId"],
                "position": board.fen(),
                "solution": moves[1],
                "solution_san": board.san(sol),
                "moves": r["Moves"],
                "rating": r["Rating"],
                "popularity": r["Popularity"],
                "nb_plays": r["NbPlays"],
                "themes": list(r["Themes"]),
            }) + "\n")
            kept += 1
    print(f"wrote {kept} puzzles -> {args.out}")


if __name__ == "__main__":
    main()
