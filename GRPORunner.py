from trl import GRPOConfig, GRPOTrainer

from rewards import get_reward_funcs
from settings import INF_EPS_HIGH, GRPO_CONFIG, HPC
from tasks import get_task
from ScheduledGRPOTrainer import ScheduledGRPOTrainer
from EntropyThermostatGRPOTrainer import EntropyThermostatGRPOTrainer


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
        if getattr(args, "entropy_thermostat", False):
            print(
                f"Entropy thermostat ON: target={args.thermostat_target}, "
                f"deadband={args.thermostat_deadband}"
            )
        else:
            print(f"Scheduling switches at step {args.switch_step}" if args.switch_step else "Scheduling OFF")
        print(f"Running on {'HPC' if HPC else 'FLAMINGO'}")
        print("="*100)

        if getattr(args, "entropy_thermostat", False):
            if args.switch_step:
                raise ValueError("--entropy_thermostat and --switch_step are mutually exclusive")
            if args.thermostat_target is None:
                raise ValueError("--entropy_thermostat requires --thermostat_target")
            if reward_name not in {"random", "random_p03", "random_p07", "gaussian"}:
                raise ValueError(
                    "The SR entropy thermostat requires a reward-independent signal: "
                    "use --reward random (or random_p03/random_p07/gaussian)."
                )

            trainer = EntropyThermostatGRPOTrainer(
                model=model,
                processing_class=tokenizer,
                args=grpo_args,
                reward_funcs=reward_funcs,
                train_dataset=ds,
                thermostat={
                    "target": float(args.thermostat_target),
                    "deadband": float(args.thermostat_deadband),
                    "ema_alpha": float(args.thermostat_ema_alpha),
                    "control_interval": int(args.thermostat_interval),
                    "up_epsilon_low": float(args.thermostat_up_eps_low),
                    "up_epsilon_high": self._parse_epsilon_high(args.thermostat_up_eps_high),
                    "down_epsilon_low": float(args.thermostat_down_eps_low),
                    "down_epsilon_high": self._parse_epsilon_high(args.thermostat_down_eps_high),
                },
            )
        elif args.switch_step: # TODO add multi-scheduling to args
            trainer = ScheduledGRPOTrainer.with_switch(
                model=model,
                args=grpo_args,
                reward_funcs=reward_funcs,      # used until the switch
                train_dataset=ds,
                switch_step=args.switch_step,
                new_epsilon=getattr(args, "new_eps_low", 0.20),                
                new_epsilon_high=getattr(args, "new_eps_high", 0.28),           
                new_reward_funcs=get_reward_funcs(getattr(args, "new_reward", "ground_truth")),  # callable or list
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
    def _parse_epsilon_high(value):
        return INF_EPS_HIGH if str(value).lower() in ("inf", "infinity") else float(value)

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
            config["vllm_gpu_memory_utilization"] = 0.275 # TEMP CHANGE # could be increased
            config["per_device_train_batch_size"] = 4 
            config["gradient_accumulation_steps"] = 16
            print("Qwen-3B, decreasing memory usage")

        if getattr(args, "dataset", "x") == "wordle":
            config["vllm_gpu_memory_utilization"] = 0.20 if HPC else 0.22

        if getattr(args, "dataset", "x") == "mbpp":
            if "Qwen" in args.model:
                config["vllm_gpu_memory_utilization"] = 0.25
            elif HPC:
                config["vllm_gpu_memory_utilization"] = 0.35
            else:
                config["vllm_gpu_memory_utilization"] = 0.40

        if getattr(args, "dataset", "x") == "dapo":
            config["vllm_gpu_memory_utilization"] = 0.35 if HPC else 0.4
            config["per_device_train_batch_size"] = 4 
            config["gradient_accumulation_steps"] = 16

        if getattr(args, "eps_low", None) is not None:
            config["epsilon"] = float(args.eps_low)
        if getattr(args, "eps_high", None) is not None:
            eh = args.eps_high
            config["epsilon_high"] = GRPORunner._parse_epsilon_high(eh)
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
