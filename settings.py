from peft import LoraConfig

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
# Qwen2.5-Math is a *base* model (no chat template). We use the standard
# math prompt and grade on the \boxed{} answer, matching the Spurious Rewards
# setup (Shao, Li et al. 2025, arXiv:2506.10947) which trains on DeepScaleR.
system_prompt = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

COMMON = dict(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    learning_rate=2e-5,
    num_train_epochs=1.5,
    logging_steps=10,
    save_steps=500,
    report_to=["wandb"],
    bf16=True,
)

# ---------------------------------------------------------------------------
# GRPO config — tuned to reproduce the spurious-reward phenomenon on Qwen-Math.
# ---------------------------------------------------------------------------
# Notes on the choices:
#   * learning_rate 1e-6 is the paper-scale LR for the 7B model. If you train
#     Qwen2.5-Math-1.5B instead (much cheaper, still shows the effect), 3e-6
#     is a reasonable bump.
#   * num_generations (G) is the GRPO group size: rollouts sampled per prompt.
#     Advantages are computed *within* the group, so you need G >= 2. This is
#     the single most important GRPO knob and was missing before.
#   * TRL constraint: the generation batch
#       (per_device_train_batch_size * grad_accum * num_processes)
#     must be divisible by num_generations. Here 2 * 8 * 1 = 16, 16 % 8 == 0,
#     i.e. 2 prompts per optimizer step.
#   * beta = 0.0 disables the KL penalty and the reference model entirely
#     (saves a lot of memory). The spurious-reward gains come from the GRPO
#     clipping bias, not from a KL anchor; set beta=1e-3 if you want a light
#     anchor for stability.
#   * Qwen2.5-Math has a 4096-token context, so keep
#     max_prompt_length + max_completion_length < 4096.
GRPO_CONFIG = dict(
    learning_rate=1e-6,             # 7B; use ~3e-6 for the 1.5B variant
    per_device_train_batch_size=2,  # small batch for GPU memory
    gradient_accumulation_steps=8,  # gen batch = 2 * 8 = 16 -> 2 prompts/step
    gradient_checkpointing=True,
    num_generations=4,              # NOTE: inbcrease later
    temperature=1.0,                # rollout sampling temperature (exploration)
    beta=0.0,                       # KL coeff; 0.0 => no reference model
    # max_prompt_length=1024,
    max_completion_length=1400,     # NOTE: increase later
    max_steps=300,                  # NOTE: increase later
    num_train_epochs=1,             # ignored once max_steps is set
    logging_steps=5,
    save_steps=250,
    bf16=True,                      # mixed precision
    report_to=["wandb"],
    max_grad_norm=1.0,

    # vllm acceleration
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.25,   # KV-cache share on the same GPU; tune this
    vllm_enable_sleep_mode=True,        # offload vLLM weights/cache during the optimizer step
    optim="paged_adamw_8bit",   # in GRPO_CONFIG; needs bitsandbytes installed
    # vllm_max_model_len=3072,          # set >= max_prompt_length + max_completion_length
)


LORA_CONFIG = LoraConfig(
    r=16,  # Rank: adaptation capacity (16 good for reasoning tasks)
    lora_alpha=32,  # Scaling factor (typically 2x rank)
    target_modules="all-linear",
    lora_dropout=0.1,  # Regularization to prevent overfitting
    bias="none",  # Skip bias adaptation for simplicity
    task_type="CAUSAL_LM",  # Causal language modeling task
)

# Debug configuration
DEBUG_EVERY = 5  # print debug info every N steps
DEBUG_N = 1  # number of samples to print per debug step
