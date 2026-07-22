from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer

from rewards import get_reward_funcs
from settings import (INF_EPS_HIGH,
                      GRPO_CONFIG, system_prompt)

GSM8K_INSTRUCTION = (
    'Let\'s think step by step and output the final answer after "####".'
)


class GRPORunner():

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

        return ds.map(_proc, remove_columns=ds.column_names)


    def run(self, model, tokenizer, args):
        
        dataset_name = getattr(args, "dataset", "gsm8k")

        if dataset_name == "gsm8k":
            ds = self.load_gsm8k_rl(tokenizer)
            default_reward = "random"
        else:  # DeepScaleR
            ds = self.convert_to_grpo(self.load_math(), tokenizer)
            default_reward = "ground_truth"

        reward_name = getattr(args, "reward", None) or default_reward
        reward_funcs = get_reward_funcs(reward_name)

        config = self.handle_grpo_config_args(args)
        self.print_config(config)
        print(f"scheduling: {args.switch}")
        grpo_args = GRPOConfig(**config)

        if args.switch:
            from scheduled_grpo_trainer import ScheduledGRPOTrainer
            trainer = ScheduledGRPOTrainer.with_switch(
                model=model,
                args=grpo_args,
                reward_funcs=reward_funcs,      # used until the switch
                train_dataset=ds,
                switch_step=args.switch_step,
                new_epsilon=0.2,                  # -> epsilon_low
                new_epsilon_high=0.28,             # -> epsilon_high
                new_reward_funcs=get_reward_funcs("ground_truth"),  # callable or list
            )
            trainer.train()

        else:
            trainer = GRPOTrainer(
                model=model,
                processing_class=tokenizer,
                reward_funcs=reward_funcs,
                args=grpo_args,
                train_dataset=ds,
            )
            trainer.train()


    # DeepScaleR 
    def convert_to_grpo(self, ds: Dataset, tokenizer) -> Dataset:
        return ds.map(
            lambda x: self.grpo_processing(x, tokenizer),
            remove_columns=ds.column_names,
            load_from_cache_file=False,
        )

    # DeepScaleR 
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


    @staticmethod
    def handle_grpo_config_args( args):

        config = dict(GRPO_CONFIG)

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
        if getattr(args, "lr", None) is not None:
            config["learning_rate"] = float(args.lr)

        # flamingo specific
        if args.model == "meta-llama/Llama-3.2-1B-Instruct":
            config["vllm_gpu_memory_utilization"] = 0.40

        config["output_dir"] = args.output_dir
        return config


    @staticmethod
    def print_config(config: dict):
        print("=" * 100)
        for key, value in config.items():
            print(f"{key}: {value}")
        print("=" * 100)