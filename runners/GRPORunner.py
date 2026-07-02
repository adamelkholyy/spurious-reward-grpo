from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer

from runners.PostTrainer import PostTrainer
from runners.rewards import get_reward_funcs
from settings import GRPO_CONFIG, system_prompt
from utils import save_model


class GRPORunner(PostTrainer):

    def load_math(self):
        """
        DeepScaleR columns: problem, solution, answer 
        """
        return load_dataset(
            "agentica-org/DeepScaleR-Preview-Dataset", split="train"
        )

    def run(self, model, tokenizer, args):
        ds = self.load_math()
        ds = self.convert_to_grpo(ds, tokenizer)

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