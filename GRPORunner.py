from trl import GRPOConfig, GRPOTrainer

from rewards import get_reward_funcs
from settings import INF_EPS_HIGH, GRPO_CONFIG, HPC
from tasks import get_task
from ScheduledGRPOTrainer import ScheduledGRPOTrainer


class GRPORunner():

    def run(self, model, tokenizer, args):
        # Resolve the dataset task (prompt formatting + rewards + gold/grading
        # all live in tasks/<name>.py). Adding a dataset needs no changes here.
        task = get_task(getattr(args, "dataset", "gsm8k"))

        ds = task.build_train(tokenizer)

        reward_name = task.resolve_reward(getattr(args, "reward", None))
        task.validate_reward(reward_name)
        reward_funcs = get_reward_funcs(reward_name)

        config = self.handle_grpo_config_args(args)
        grpo_args = GRPOConfig(**config)

        self.print_config(config)
        print(f"Scheduling switches at step {args.switch_step}" if args.switch_step else "Scheduling OFF")
        print(f"Running on {'HPC' if HPC else 'FLAMINGO'}")
        print("="*100)

        if args.switch_step: # TODO add multi-scheduling to args
            trainer = ScheduledGRPOTrainer.with_switch(
                model=model,
                args=grpo_args,
                reward_funcs=reward_funcs,      # used until the switch
                train_dataset=ds,
                switch_step=args.switch_step,
                new_epsilon=getattr(args, "new_eps_low", 0.20),                
                new_epsilon_high=getattr(args, "new_eps_high", 0.28),           
                new_reward_funcs=get_reward_funcs("ground_truth"),  # callable or list
            )
        else:
            trainer = GRPOTrainer(
                model=model,
                processing_class=tokenizer,
                reward_funcs=reward_funcs,
                args=grpo_args,
                train_dataset=ds,
            )
        trainer.train()

    @staticmethod
    def handle_grpo_config_args(args):

        config = dict(GRPO_CONFIG)

        if HPC: # CSD3 specific settings
            if "Qwen" in  args.model:
                config["vllm_gpu_memory_utilization"] = 0.22
                print("Qwen, decreasing memory usage")
        else: # Flamingo specific settings 
            if "0.5B" in args.model:
                config["vllm_gpu_memory_utilization"] = 0.5
                print("Qwen-0.5B, increasing memory usage")

        if "3B" in args.model:
            config["vllm_gpu_memory_utilization"] = 0.275 # could be increased
            config["per_device_train_batch_size"] = 4 
            config["gradient_accumulation_steps"] = 16
            print("Qwen-3B, decreasing memory usage")

        if getattr(args, "dataset", "x") == "wordle":
            config["vllm_gpu_memory_utilization"] = 0.22

        if getattr(args, "dataset", "x") == "mbpp":
            config["vllm_gpu_memory_utilization"] = 0.5

        if getattr(args, "dataset", "x") == "dapo":
            config["vllm_gpu_memory_utilization"] = 0.5
            config["per_device_train_batch_size"] = 4 
            config["gradient_accumulation_steps"] = 16

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



if __name__ == "__main__":
    # Minimal smoke test of the scheduling logic (no real training run).
    # Demonstrates the intended usage without needing a GPU/model.
    def reward_phase_a(completions, **kwargs):
        return [float(len(c)) for c in completions]

    def reward_phase_b(completions, **kwargs):
        return [-abs(200 - len(c)) for c in completions]

    print("Example schedule (pass to ScheduledGRPOTrainer(schedule=...)):")
    example_schedule = [
        {"step": 0, "epsilon": 0.2, "reward_funcs": reward_phase_a},
        {
            "step": 500,
            "epsilon": 0.3,
            "epsilon_high": 0.4,
            "reward_funcs": reward_phase_b,
        },
    ]
    for c in example_schedule:
        print("  ", c)