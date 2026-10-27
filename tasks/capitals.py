"""Capitals task: a replication of one NoveltyBench (NB-Curated) prompt.

NoveltyBench (Zhang et al., 2025, arXiv:2504.05228) measures how many
meaningfully different answers a model gives when it is asked an underspecified
question with many valid answers. One of the NB-Curated "factual knowledge"
prompts asks the model to name a capital city in Africa. NoveltyBench draws
k=10 samples at temperature 1.0 and counts equivalence classes (``distinct_k``)
with a fine-tuned DeBERTa classifier.

Here the equivalence class is simply the *country* the completion names, so the
metric is exact: ``distinct_countries`` = the number of different African
countries named across the draws (use ``eval.py --n 100`` for 100 draws).
How evenly the draws spread over those countries is reported alongside it as
``effective_number`` (inverse Simpson D2) and ``evenness`` (D2 / distinct);
see ``spread_metrics``.

Training uses a single prompt, repeated ``n_train`` times to give the trainer
something to iterate over (``n_train`` is a class attribute; with identical rows
its value only matters for ``--data_fraction`` arithmetic, since ``--max_steps``
sets the run length).

Reward (``capitals_diversity``): diversity is a property of a GROUP of
completions to the same prompt, so the reward is computed per group of
``num_generations`` consecutive completions (the group size is handed over by
``GRPORunner`` through ``rewards.set_num_generations``). Each completion earns

    1 / (number of completions in its group naming the same country)

and 0 if it names no recognisable African country. The rewards in a group
therefore sum to that group's distinct-country count, and rarer answers earn
more than common ones.

Caveats:
  * The country is the FIRST African country / capital / alias mentioned in the
    completion (earliest position; longest alias wins ties, so "Guinea-Bissau"
    is not "Guinea").
  * A bare "Congo" is ambiguous (two countries) and is not matched; use
    "Brazzaville", "Kinshasa", "DR Congo", etc.
  * Under multi-GPU training, rewards are computed per process on its local
    slice. Groups are only exact if each process's slice holds whole groups
    (local completions divisible by ``num_generations``); a trailing partial
    chunk is treated as its own group.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Sequence

from .base import DatasetSpec
from .registry import register_task

PROMPT = "Name a capital city in Africa."

# Canonical country -> extra aliases (capitals, de-facto/legislative capitals,
# alternate names). The country's own name is always an alias. 54 UN member
# states. Written without diacritics; input text is normalised the same way.
AFRICAN_COUNTRIES: Dict[str, List[str]] = {
    "Algeria": ["Algiers"],
    "Angola": ["Luanda"],
    "Benin": ["Porto-Novo", "Porto Novo", "Cotonou"],
    "Botswana": ["Gaborone"],
    "Burkina Faso": ["Ouagadougou"],
    "Burundi": ["Gitega", "Bujumbura"],
    "Cabo Verde": ["Cape Verde", "Praia"],
    "Cameroon": ["Yaounde"],
    "Central African Republic": ["Bangui"],
    "Chad": ["N'Djamena", "Ndjamena"],
    "Comoros": ["Moroni"],
    "Democratic Republic of the Congo": [
        "Democratic Republic of Congo", "DR Congo", "D.R. Congo", "DRC",
        "Congo-Kinshasa", "Kinshasa",
    ],
    "Republic of the Congo": ["Republic of Congo", "Congo-Brazzaville", "Brazzaville"],
    "Cote d'Ivoire": ["Ivory Coast", "Yamoussoukro", "Abidjan"],
    "Djibouti": [],
    "Egypt": ["Cairo"],
    "Equatorial Guinea": ["Malabo", "Ciudad de la Paz"],
    "Eritrea": ["Asmara"],
    "Eswatini": ["Swaziland", "Mbabane", "Lobamba"],
    "Ethiopia": ["Addis Ababa"],
    "Gabon": ["Libreville"],
    "Gambia": ["The Gambia", "Banjul"],
    "Ghana": ["Accra"],
    "Guinea": ["Conakry"],
    "Guinea-Bissau": ["Bissau"],
    "Kenya": ["Nairobi"],
    "Lesotho": ["Maseru"],
    "Liberia": ["Monrovia"],
    "Libya": ["Tripoli"],
    "Madagascar": ["Antananarivo"],
    "Malawi": ["Lilongwe"],
    "Mali": ["Bamako"],
    "Mauritania": ["Nouakchott"],
    "Mauritius": ["Port Louis"],
    "Morocco": ["Rabat"],
    "Mozambique": ["Maputo"],
    "Namibia": ["Windhoek"],
    "Niger": ["Niamey"],
    "Nigeria": ["Abuja"],
    "Rwanda": ["Kigali"],
    "Sao Tome and Principe": ["Sao Tome"],
    "Senegal": ["Dakar"],
    "Seychelles": [],  # capital "Victoria" handled by _VICTORIA_RE below
    "Sierra Leone": ["Freetown"],
    "Somalia": ["Mogadishu"],
    "South Africa": ["Pretoria", "Cape Town", "Bloemfontein"],
    "South Sudan": ["Juba"],
    "Sudan": ["Khartoum"],
    "Tanzania": ["Dodoma", "Dar es Salaam"],
    "Togo": ["Lome"],
    "Tunisia": ["Tunis"],
    "Uganda": ["Kampala"],
    "Zambia": ["Lusaka"],
    "Zimbabwe": ["Harare"],
}


# ---------------------------------------------------------------------------
# Country extraction
# ---------------------------------------------------------------------------
def _normalise(text: str) -> str:
    """Lowercase, strip diacritics, straighten apostrophes."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower().replace("’", "'").replace("‘", "'").replace("`", "'")


def _build_alias_table() -> Dict[str, str]:
    table: Dict[str, str] = {}
    for country, aliases in AFRICAN_COUNTRIES.items():
        for alias in [country, *aliases]:
            key = _normalise(alias)
            if key in table and table[key] != country:
                raise ValueError(f"Alias '{alias}' maps to two countries")
            table[key] = country
    return table


_ALIAS_TO_COUNTRY = _build_alias_table()

# Longest alias first: at any start position the regex then prefers the longest
# alias, and the letter lookarounds stop "niger" matching inside "nigeria".
_ALIAS_RE = re.compile(
    r"(?<![a-z])("
    + "|".join(re.escape(a) for a in sorted(_ALIAS_TO_COUNTRY, key=len, reverse=True))
    + r")(?![a-z])"
)

# "Victoria" is the Seychelles capital but also a lake, a waterfall, a queen...
_VICTORIA_RE = re.compile(
    r"(?<![a-z])(?<!lake )(?<!queen )victoria(?![a-z])(?!\s+(?:falls|island|nile|lake|land))"
)


def extract_country(completion: str) -> Optional[str]:
    """The first African country named in ``completion`` (by country name,
    capital or alias), or ``None`` if there is none."""
    text = _normalise(completion)
    best_pos, best = None, None

    m = _ALIAS_RE.search(text)
    if m:
        best_pos, best = m.start(), _ALIAS_TO_COUNTRY[m.group(1)]

    v = _VICTORIA_RE.search(text)
    if v and (best_pos is None or v.start() < best_pos):
        best = "Seychelles"

    return best


def country_histogram(completions: Sequence[str]) -> Counter:
    """Counter of countries named across ``completions`` (unmatched ignored)."""
    return Counter(c for c in map(extract_country, completions) if c)


def distinct_countries(completions: Sequence[str]) -> int:
    """The headline metric: how many different countries were named."""
    return len(country_histogram(completions))


def spread_metrics(counts: Dict[str, int]) -> dict:
    """How evenly the draws are spread over the countries named.

    ``effective_number`` is the inverse Simpson index D2 = 1 / sum(p_i^2): the
    number of equally common countries with the same diversity (1/D2 is the
    probability two draws name the same country). ``evenness`` is D2 / K in
    (0, 1], 1 iff every named country is equally common. Both use plug-in
    frequencies, so D2 <= K. Compare evenness at equal valid sample sizes: K
    grows with N, D2 barely does.
    """
    n = sum(counts.values())
    if n == 0:
        return {"effective_number": 0.0, "evenness": 0.0}
    d2 = 1.0 / sum((c / n) ** 2 for c in counts.values())
    return {"effective_number": d2, "evenness": d2 / len(counts)}


def hill_numbers(counts: Dict[str, int], qs: Sequence[float]) -> List[float]:
    """Hill numbers D_q = (sum p_i^q)^(1 / (1 - q)) of the draws, one per order q.

    D_0 is the distinct count, D_1 = exp(Shannon entropy) (the q -> 1 limit), D_2 is
    ``spread_metrics``' effective_number; larger q weights common answers more.
    """
    n = sum(counts.values())
    if n == 0:
        return [0.0 for _ in qs]
    ps = [c / n for c in counts.values() if c > 0]
    out = []
    for q in qs:
        if abs(q - 1.0) < 1e-9:
            out.append(math.exp(-sum(p * math.log(p) for p in ps)))
        else:
            out.append(sum(p**q for p in ps) ** (1.0 / (1.0 - q)))
    return out


# ---------------------------------------------------------------------------
# Reward — registered as 'capitals_diversity'
# ---------------------------------------------------------------------------
def group_rewards(countries: Sequence[Optional[str]]) -> List[float]:
    """Inverse-frequency reward for ONE group of per-completion countries.

    ``1/n`` for a completion whose country appears ``n`` times in the group,
    ``0.0`` for no recognised country. Sums to the group's distinct count.
    """
    counts = Counter(c for c in countries if c is not None)
    return [1.0 / counts[c] if c is not None else 0.0 for c in countries]


def capitals_diversity_reward(prompts, completions, **kwargs):
    """Per-group inverse-frequency diversity reward (see module docstring)."""
    from rewards import get_num_generations
    from utils import get_completion_text

    responses = [get_completion_text(c) for c in completions]
    countries = [extract_country(r) for r in responses]

    g = get_num_generations() or len(countries)  # unset => one big group
    rewards: List[float] = []
    distinct_per_group: List[int] = []
    for i in range(0, len(countries), g):
        chunk = countries[i : i + g]
        rewards.extend(group_rewards(chunk))
        distinct_per_group.append(len({c for c in chunk if c}))

    try:
        import wandb

        if wandb.run is not None and distinct_per_group:
            wandb.log(
                {
                    "train/capitals_distinct_per_group": sum(distinct_per_group)
                    / len(distinct_per_group),
                    "train/capitals_valid_frac": sum(c is not None for c in countries)
                    / len(countries),
                },
                commit=False,
            )
    except Exception:
        pass

    return rewards


def _register_reward():
    from rewards import REWARD_REGISTRY

    REWARD_REGISTRY.setdefault("capitals_diversity", [capitals_diversity_reward])


_register_reward()


# ---------------------------------------------------------------------------
# Task spec
# ---------------------------------------------------------------------------
@register_task
class CapitalsTask(DatasetSpec):
    name = "capitals"
    hf_path = "capitals"   # no hub dataset: informational only
    hf_config = None
    train_split = "train"
    eval_split = "train"

    default_reward = "capitals_diversity"
    allowed_rewards = ("capitals_diversity", "random")

    system_prompt = None
    train_instruction = None

    #: Number of (identical) training rows.
    n_train: int = 10_000

    # ---- training ----
    def format_train_example(self, x, tokenizer):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": prompt, "answer": ""}

    def build_train(self, tokenizer):
        from datasets import Dataset

        row = self.format_train_example({}, tokenizer)
        return Dataset.from_dict(
            {
                "prompt": [row["prompt"]] * self.n_train,
                "answer": [row["answer"]] * self.n_train,
            }
        )

    # ---- eval ----
    def eval_dataset_id(self, split=None) -> str:
        return "capitals:african_capital"

    def load_eval(self, split=None, limit=None):
        return [PROMPT], [""]

    def sample_questions(self, split, num_prompts, seed):
        return [PROMPT] * num_prompts

    def extract_gold(self, answer_field) -> str:
        return ""

    def extract_answer(self, completion: str) -> Optional[str]:
        return extract_country(completion)

    def make_grader(self):
        """Per-sample flag: did the draw name any recognisable African
        country? (So eval.py's ``accuracy`` is the *valid-answer rate*; the
        diversity number lives in ``summarise_samples``.)"""

        def _names_a_country(completion: str, gold) -> bool:
            return extract_country(completion) is not None

        return _names_a_country

    def summarise_samples(self, samples_per_problem: Sequence[Sequence[str]]) -> dict:
        """Diversity metrics over the draws for the (single) eval prompt.
        Called by eval.py with one list of completions per problem."""
        texts = list(samples_per_problem[0])
        hist = country_histogram(texts)
        return {
            "distinct_countries": len(hist),
            "n_samples": len(texts),
            "n_valid": sum(hist.values()),
            **spread_metrics(hist),
            "country_counts": dict(hist.most_common()),
        }