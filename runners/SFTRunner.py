from datasets import Dataset, load_dataset
from trl import SFTConfig, SFTTrainer

from backups.runners.PostTrainer import PostTrainer
from backups.settings import COMMON, system_prompt
from backups.utils import save_model, strip_calculator_annotations


class SFTRunner(PostTrainer):

    # custom dataset loading for SFT
    def load_gsm8k(self) -> Dataset:
        ds = load_dataset("openai/gsm8k", "main", split="train")
        cols_to_remove = [
            c for c in ds.column_names if c not in ("prompt", "completion")
        ]
        return ds.map(
            self.format_gsm8k, remove_columns=cols_to_remove, load_from_cache_file=False
        )

    @staticmethod
    def format_gsm8k(x) -> dict:
        """Format as prompt/completion for completion-only SFT."""
        prompt = f"{x['question']}\n{system_prompt}"
        answer = strip_calculator_annotations(x["answer"])
        completion = f" {answer}"  # leading space so it tokenises nicely
        return {"prompt": prompt, "completion": completion}

    def run(self, model, _tokenizer, args):
        ds = self.load_gsm8k()

        config = dict(
            COMMON,
            num_train_epochs=3, # STO specific
            output_dir=args.output_dir,
        )
        self.print_config(config)

        trainer = SFTTrainer(
            model=model,
            train_dataset=ds,
            #callbacks=[WandbCallback()],
            args=SFTConfig(**config),
        )
        trainer.train()
        save_model(trainer, "sft")
