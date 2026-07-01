
export SCRATCH=/local/scratch/$USER

export HF_HOME=$SCRATCH/.cache/huggingface
export HF_DATASETS_CACHE=$SCRATCH/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=$SCRATCH/.cache/huggingface/transformers
export HUGGINGFACE_HUB_CACHE=$SCRATCH/.cache/huggingface/hub

export WANDB_DIR=$SCRATCH/wandb
export WANDB_CACHE_DIR=$SCRATCH/.cache/wandb
export WANDB_CONFIG_DIR=$SCRATCH/.config/wandb
export WANDB_API_KEY=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
export HOME=/local/scratch/ae581
export XDG_CACHE_HOME=/local/scratch/ae581/.cache
export FLASHINFER_WORKSPACE_DIR=/local/scratch/ae581/.cache/flashinfer
export VLLM_CACHE_ROOT=/local/scratch/ae581/.cache/vllm


# CUDA_VISIBLE_DEVICES=1 python eval_math500.py --models base=Qwen/Qwen2.5-Math-1.5B # gt=outputs/grpo_base-1782733692/checkpoint-grpo_ground_truth
# CUDA_VISIBLE_DEVICES=1 accelerate launch trainer.py --reward random --run_name grpo_random

CUDA_VISIBLE_DEVICES=1 python eval_math500.py --models base=Qwen/Qwen2.5-Math-1.5B gt=outputs/grpo_random_qwen1.5/checkpoint-grpo_random
 