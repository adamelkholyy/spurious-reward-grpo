import os

from datasets import concatenate_datasets
from trl import GRPOConfig, GRPOTrainer

from rewards import get_reward_funcs, set_num_generations
from settings import INF_EPS_HIGH, GRPO_CONFIG, HPC
from tasks import get_task
from ScheduledGRPOTrainer import ScheduledGRPOTrainer


def run(model, tokenizer, args):
    task = get_task(args.dataset)
    ds = task.build_train(tokenizer)

    # optionally train on a random subset of the data
    if args.single_example:
        ds = ds.shuffle(seed=args.seed).select([0])
        print(f"Using a single training example (single_example, seed={args.seed})")
    elif args.num_examples is not None:
        ds = ds.shuffle(seed=args.seed).select(range(min(args.num_examples, len(ds))))
        print(f"Using {len(ds)} training examples (num_examples, seed={args.seed})")
    elif args.data_fraction < 1.0:
        full_size = len(ds)
        n = max(1, round(full_size * args.data_fraction))
        ds = ds.shuffle(seed=args.seed).select(range(n))
        print(f"Using {n}/{full_size} training examples (data_fraction={args.data_fraction})")

    reward_name = task.resolve_reward(args.reward)
    task.validate_reward(reward_name)
    reward_funcs = get_reward_funcs(reward_name)

    config = build_grpo_config(args)
    grpo_args = GRPOConfig(**config)
    set_num_generations(grpo_args.num_generations)  # for group-level rewards

    # TRL's RepeatSampler drops any incomplete chunk of generation_batch_size // num_generations
    # prompts, so a subset smaller than one chunk gives an empty epoch and training stops at step 0.
    # Repeat it to fill a chunk: still only these n distinct problems, each seen more often.
    prompts_per_gen = grpo_args.generation_batch_size // grpo_args.num_generations
    if len(ds) < prompts_per_gen:
        reps = -(-prompts_per_gen // len(ds))  # ceil
        print(f"Only {len(ds)} examples but each generation batch takes {prompts_per_gen} prompts: "
              f"repeating the dataset {reps}x")
        ds = concatenate_datasets([ds] * reps)

    print("=" * 100)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 100)
    print(f"Scheduling switches at step {args.switch_step}" if args.switch_step else "Scheduling OFF")
    print(f"Running on {'HPC' if HPC else 'FLAMINGO'}")
    print("=" * 100)

    if args.switch_step:
        trainer = ScheduledGRPOTrainer.with_switch(
            model=model,
            processing_class=tokenizer,
            args=grpo_args,
            reward_funcs=reward_funcs,  # used until the switch
            train_dataset=ds,
            switch_step=args.switch_step,
            new_epsilon=args.new_eps_low,
            new_epsilon_high=args.new_eps_high,
            new_reward_funcs=get_reward_funcs(args.new_reward),
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


def parse_epsilon_high(value):
    return INF_EPS_HIGH if str(value).lower() in ("inf", "infinity") else float(value)


def build_grpo_config(args):
    config = dict(GRPO_CONFIG)

    # vLLM memory / batch overrides per machine, model and dataset; later overrides win
    if HPC:  # CSD3
        if "qwen" in args.model.lower():
            config["vllm_gpu_memory_utilization"] = 0.22
            print("Qwen, decreasing memory usage")
    else:  # Flamingo
        if "0.5B" in args.model:
            config["vllm_gpu_memory_utilization"] = 0.5
            print("Qwen-0.5B, increasing memory usage")

    if "3b" in args.model.lower():
        config["vllm_gpu_memory_utilization"] = 0.275
        config["per_device_train_batch_size"] = 4
        config["gradient_accumulation_steps"] = 16
        print("Qwen-3B, decreasing memory usage")

    if args.dataset == "wordle":
        config["vllm_gpu_memory_utilization"] = 0.20 if HPC else 0.22

    if args.dataset == "mbpp":
        if "qwen" in args.model.lower():
            config["vllm_gpu_memory_utilization"] = 0.25
        elif HPC:
            config["vllm_gpu_memory_utilization"] = 0.35
        else:
            config["vllm_gpu_memory_utilization"] = 0.40

    if args.dataset == "dapo":
        config["vllm_gpu_memory_utilization"] = 0.35 if HPC else 0.4
        config["per_device_train_batch_size"] = 4
        config["gradient_accumulation_steps"] = 16

    config["epsilon"] = args.eps_low
    config["epsilon_high"] = parse_epsilon_high(args.eps_high)
    config["max_steps"] = args.max_steps
    config["seed"] = args.seed
    if args.lr is not None:
        config["learning_rate"] = args.lr
    if args.save_step is not None:
        config["save_steps"] = args.save_step

    if args.model == "meta-llama/Llama-3.2-1B-Instruct":
        config["vllm_gpu_memory_utilization"] = 0.40

    # keep the global batch fixed under multi-GPU; if GRPO_CONFIG sets
    # steps_per_generation / generation_batch_size explicitly, scale those too
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        ga = config["gradient_accumulation_steps"]
        assert ga % world_size == 0, f"grad accum {ga} not divisible by {world_size} GPUs"
        config["gradient_accumulation_steps"] = ga // world_size

    config["output_dir"] = args.output_dir
    return config
