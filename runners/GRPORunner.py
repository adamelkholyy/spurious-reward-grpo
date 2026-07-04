from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer

from runners.PostTrainer import PostTrainer
from runners.rewards import get_reward_funcs
from settings import GRPO_CONFIG, INF_EPS_HIGH, PARK_GRPO_CONFIG, system_prompt
from utils import save_model

# verl's exact GSM8K instruction suffix (examples/data_preprocess/gsm8k.py),
# which Park et al. (arXiv:2509.26114) use via the verl GSM8K recipe. It is
# appended to the question inside a single user turn — no system message.
GSM8K_INSTRUCTION = (
    'Let\'s think step by step and output the final answer after "####".'
)


class GRPORunner(PostTrainer):

    # ------------------------------------------------------------------ data
    def load_math(self):
        """DeepScaleR columns: problem, solution, answer (legacy Shao-style)."""
        return load_dataset(
            "agentica-org/DeepScaleR-Preview-Dataset", split="train"
        )

    def load_gsm8k_rl(self, tokenizer) -> Dataset:
        """GSM8K train formatted exactly like verl's GSM8K recipe:
        user message = question + ' ' + instruction, chat template applied,
        gold answer = text after '####' with commas/'$' stripped."""
        ds = load_dataset("openai/gsm8k", "main", split="train")

        def _proc(x):
            question = f"{x['question']} {GSM8K_INSTRUCTION}"
            msgs = [{"role": "user", "content": question}]
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            gold = x["answer"].split("####")[-1].strip()
            gold = gold.replace(",", "").replace("$", "")
            return {"prompt": prompt, "answer": gold}

        return ds.map(_proc, remove_columns=ds.column_names,
                      load_from_cache_file=False)

    # ------------------------------------------------------------------- run
    def run(self, model, tokenizer, args):
        dataset_name = getattr(args, "dataset", "gsm8k")

        if dataset_name == "gsm8k":
            ds = self.load_gsm8k_rl(tokenizer)
            config = dict(PARK_GRPO_CONFIG)
            default_reward = "random"
        else:  # deepscaler (legacy Shao-style setup)
            ds = self.convert_to_grpo(self.load_math(), tokenizer)
            config = dict(GRPO_CONFIG)
            default_reward = "ground_truth"

        reward_name = getattr(args, "reward", None) or default_reward
        reward_funcs = get_reward_funcs(reward_name)

        # ---- clip-parameter overrides (Park et al. ablations) ----
        # eps_low:  1.0   -> clip-low OFF   (their Fig. 3 / Fig. 5 setting)
        # eps_high: "inf" -> clip-high OFF  (mapped to INF_EPS_HIGH)
        if getattr(args, "eps_low", None) is not None:
            config["epsilon"] = float(args.eps_low)
        if getattr(args, "eps_high", None) is not None:
            eh = args.eps_high
            config["epsilon_high"] = (
                INF_EPS_HIGH if str(eh).lower() in ("inf", "infinity")
                else float(eh)
            )
        if getattr(args, "max_steps", None) is not None:
            config["max_steps"] = int(args.max_steps)
        if getattr(args, "seed", None) is not None:
            config["seed"] = int(args.seed)

        config["output_dir"] = args.output_dir

        eps_low = config.get("epsilon", 0.2)
        eps_high = config.get("epsilon_high")
        print(
            f"Dataset: {dataset_name} | Reward: {reward_name} -> "
            f"{[f.__name__ for f in reward_funcs]}\n"
            f"Clipping: eps_low={eps_low}, "
            f"eps_high={'symmetric (=eps_low)' if eps_high is None else eps_high}"
            f"{'  [clip-high OFF]' if eps_high == INF_EPS_HIGH else ''}"
            f"{'  [clip-low OFF]' if eps_low >= 1.0 else ''}"
        )
        self.print_config(config)

        grpo_args = GRPOConfig(**config)

        # ---- geometry sanity check (reads what TRL DERIVED, not what we set).
        # Guards against the steps_per_generation micro-step unit trap: if the
        # rollout buffer doesn't span multiple optimizer steps, pi_old is never
        # tracked, the ratio is identically 1, and clipping can never fire.
        upd_per_rollout = grpo_args.steps_per_generation // grpo_args.gradient_accumulation_steps
        print(
            f"[geometry] generation_batch={grpo_args.generation_batch_size} seqs "
            f"({grpo_args.generation_batch_size // grpo_args.num_generations} prompts x "
            f"{grpo_args.num_generations}), optimizer batch="
            f"{grpo_args.per_device_train_batch_size * grpo_args.gradient_accumulation_steps} seqs, "
            f"updates/rollout={upd_per_rollout}"
        )
        if dataset_name == "gsm8k":
            expected = dict(gen=4096, prompts=512, updates=16)
            actual = dict(
                gen=grpo_args.generation_batch_size,
                prompts=grpo_args.generation_batch_size // grpo_args.num_generations,
                updates=upd_per_rollout,
            )
            if actual != expected:
                raise ValueError(
                    f"Park et al. geometry mismatch: expected {expected}, got {actual}. "
                    "Clipping will not bind as in the paper — fix PARK_GRPO_CONFIG."
                )
        if upd_per_rollout <= 1 and grpo_args.num_iterations == 1:
            raise ValueError(
                "Rollout buffer spans <=1 optimizer step: training is fully "
                "on-policy, ratio==1, clipping can never fire. Increase "
                "generation_batch_size."
            )

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_funcs,
            args=grpo_args,
            train_dataset=ds,
        )
        trainer.train()
        save_model(trainer, f"grpo_{dataset_name}_{reward_name}")

    # -------------------------------------------------- legacy DeepScaleR fmt
    def convert_to_grpo(self, ds: Dataset, tokenizer) -> Dataset:
        return ds.map(
            lambda x: self.grpo_processing(x, tokenizer),
            remove_columns=ds.column_names,
            load_from_cache_file=False,
        )

    @staticmethod
    def grpo_processing(x, tokenizer):
        question = x["problem"]
        answer = str(x["answer"]).strip()
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt, "answer": answer}
