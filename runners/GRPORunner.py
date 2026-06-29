from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer

from backups.runners.PostTrainer import PostTrainer
from backups.runners.rewards import get_reward_funcs
from backups.settings import GRPO_CONFIG, system_prompt
from backups.utils import save_model


class GRPORunner(PostTrainer):

    # ---- Data -------------------------------------------------------------
    def load_math(self):
        """DeepScaleR (the dataset used in Spurious Rewards).

        Columns: problem, solution, answer  (answer is a clean string, e.g. "35").
        Swap for a MATH train split if you prefer, e.g.
        load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train").
        """
        return load_dataset(
            "agentica-org/DeepScaleR-Preview-Dataset", split="train"
        )

    def run(self, model, tokenizer, args):
        ds = self.load_math()
        ds = self.convert_to_grpo(ds)

        reward_name = getattr(args, "reward", "ground_truth")
        reward_funcs = get_reward_funcs(reward_name)

        config = dict(GRPO_CONFIG, output_dir=args.output_dir)
        print(f"Reward: {reward_name}  ->  {[f.__name__ for f in reward_funcs]}")
        self.print_config(config)

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_funcs,
            args=GRPOConfig(**config),
            train_dataset=ds,
        )
        trainer.train()
        save_model(trainer, f"grpo_{reward_name}")

    # ---- Preprocessing ----------------------------------------------------
    def convert_to_grpo(self, ds: Dataset) -> Dataset:
        return ds.map(
            self.grpo_processing,
            remove_columns=ds.column_names,
            load_from_cache_file=False,
        )

    @staticmethod
    def grpo_processing(x):
        """DeepScaleR example -> GRPO format (plain-text prompt + gold answer)."""
        question = x["problem"]
        answer = str(x["answer"]).strip()

        # plain-text prompt (Qwen2.5-Math is a base model, no chat template)
        prompt = f"{question}\n{system_prompt}"

        return {
            "prompt": prompt,
            "answer": answer,  # gold final answer for the reward functions
        }
