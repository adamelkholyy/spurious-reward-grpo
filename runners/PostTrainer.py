from abc import abstractmethod

from datasets import Dataset, load_dataset
from utils import strip_calculator_annotations

class PostTrainer:

    def load_gsm8k(self) -> Dataset:
        ds = load_dataset("openai/gsm8k", "main", split="train")
        cols_to_remove = [c for c in ds.column_names if c != "text"]
        return ds.map(
            self.format_gsm8k, remove_columns=cols_to_remove, load_from_cache_file=False
        )

    @staticmethod
    def format_gsm8k(x: dict) -> dict:
        # single text field used for DPO, KTO, RewardTrainer
        answer = strip_calculator_annotations(x["answer"])
        return {"text": f"Question: {x['question']}\nAnswer: {answer}"}

    @staticmethod
    def print_config(config: dict):
        print("=" * 100)
        for key, value in config.items():
            print(f"{key}: {value}")
        print("=" * 100)

    @abstractmethod
    def run(self, model, tokenizer, args):
        pass

    
