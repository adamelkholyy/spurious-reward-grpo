import os
import json

OUTPUTS_DIR = "./outputs"
RESULTS_DIR = "./results"
DISABLED_FILE = ".disabled_models.json"

RESULT_FILES = [
    "results_entropy_gsm8k.json",
    "results_gsm8k.json",
    "results_output_dist.json",
]


def load_json(path, default):
    if not os.path.exists(path):
        return default

    with open(path, "r") as f:
        return json.load(f)


def load_disabled():
    return set(load_json(DISABLED_FILE, []))


def save_disabled(disabled):
    with open(DISABLED_FILE, "w") as f:
        json.dump(sorted(disabled), f, indent=2)


def load_evaluated_models():
    """
    Returns all labels and model paths that have already appeared
    in any result file.
    """
    evaluated_labels = set()
    evaluated_paths = set()

    for filename in RESULT_FILES:
        path = os.path.join(RESULTS_DIR, filename)

        for result in load_json(path, []):
            if "label" in result:
                evaluated_labels.add(result["label"])

            if "model" in result:
                evaluated_paths.add(result["model"])

    return evaluated_labels, evaluated_paths


def get_available_models():
    """
    Models in outputs that have not already been evaluated.
    """
    evaluated_labels, evaluated_paths = load_evaluated_models()

    models = []

    for name in sorted(os.listdir(OUTPUTS_DIR)):
        path = os.path.join(OUTPUTS_DIR, name)

        if not os.path.isdir(path):
            continue

        # Skip if the run label exists in results
        if name in evaluated_labels:
            continue

        # Skip if any checkpoint from this run exists in results
        already_done = any(
            model_path.startswith(path)
            for model_path in evaluated_paths
        )

        if already_done:
            continue

        models.append(name)

    return models


def main():
    disabled = load_disabled()
    models = get_available_models()

    if not models:
        print("No unevaluated models found.")
        return

    while True:
        print("\nUnevaluated models:")
        print("-" * 80)

        for i, model in enumerate(models):
            status = "DISABLED" if model in disabled else "ACTIVE"
            print(f"{i:3d}: [{status:8s}] {model}")

        print("-" * 80)
        print("Toggle with number")
        print("[a] disable all")
        print("[e] enable all")
        print("[s] save")
        print("[q] quit")

        choice = input("> ").strip()

        if choice == "q":
            break

        elif choice == "s":
            save_disabled(disabled)
            print(f"Saved {len(disabled)} disabled models")

        elif choice == "a":
            disabled.update(models)

        elif choice == "e":
            for m in models:
                disabled.discard(m)

        else:
            try:
                idx = int(choice)
                model = models[idx]

                if model in disabled:
                    disabled.remove(model)
                else:
                    disabled.add(model)

            except (ValueError, IndexError):
                print("Invalid option")


if __name__ == "__main__":
    main()