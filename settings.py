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


####### Shao paper settings #######
# learning_rate=5e-7,             
# lr_scheduler_type="constant",    
# per_device_train_batch_size=16,  
# gradient_accumulation_steps=8,  # gen batch = 16*8 = 128 / G=16 -> 8 prompts/step
# num_generations=16,    


####### Chen paper settings #######
# learning_rate=5e-7,             # Chen: 5e-7 constant — already matched
# per_device_train_batch_size=16, # keep
# gradient_accumulation_steps=8,  # gen batch 128 -> 8 prompts/step (see caveat)
# lr_scheduler_type="constant",   # Chen holds LR flat — already matched
# num_generations=16,             # Chen: G=16 — already matched
# epsilon=0.2,                    # ADD — Chen uses symmetric clip eps=0.2
# max_completion_length=3072,     # was 2048 — Chen uses ~4096 rollout on Qwen-Math;
#                                 # 1024+3072=4096 = the context window
### vllm_max_model_len=4096,        # ADD — must cover prompt+completion

###### prev default settings #####
# learning_rate=3e-6,              
# per_device_train_batch_size=8,   
# gradient_accumulation_steps=8,  
# gradient_checkpointing=True,
# num_generations=8,               
# temperature=1.0,                 
# beta=0.0,                       # KL coeff = 0.0 -> no reference model
# max_completion_length=2048,     
# max_steps=300,        

GRPO_CONFIG = dict(
    learning_rate=5e-7,             # Chen: 5e-7 constant — already matched
    per_device_train_batch_size=8, # keep
    gradient_accumulation_steps=8,  # gen batch 128 -> 8 prompts/step (see caveat)
    lr_scheduler_type="constant",   # Chen holds LR flat — already matched
    num_generations=16,             # Chen: G=16 — already matched
    epsilon=0.2,                    # ADD — Chen uses symmetric clip eps=0.2
    temperature=1.0,                 
    beta=0.0,                       # KL coeff = 0.0 -> no reference model
    max_completion_length=2048,     
    max_steps=300,                  
    num_train_epochs=1,             # ignored once max_steps is set
    logging_steps=5,
    save_steps=250,
    bf16=True,                      # mixed precision
    report_to=["wandb"],
    max_grad_norm=1.0,

    # vllm acceleration
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.25,   # KV-cache share on same GPU
    vllm_enable_sleep_mode=True,        # offload vLLM weights/cache during the optimizer step
    optim="paged_adamw_8bit",   
    # vllm_max_model_len=3072,          # set = max_prompt_length + max_completion_length
)


LORA_CONFIG = LoraConfig(
    r=16,  # rank: adaptation capacity  
    lora_alpha=32,  # scaling factor is typically 2x rank
    target_modules="all-linear",
    lora_dropout=0.1,   
    bias="none",  # skip bias adaptation for simplicity
    task_type="CAUSAL_LM",  # causal language modeling task
)

# Debug configuration
DEBUG_EVERY = 5  # print debug info every N steps
DEBUG_N = 1  # number of samples to print per debug step
