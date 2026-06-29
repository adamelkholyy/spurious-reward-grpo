from transformers import TrainerCallback
import wandb

class WandbCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        wandb.log(logs, step=state.global_step)