import sys
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine
import data_loader
import sticky_config as config


def main():
    print(f"Loading pure HuggingFace model (no sticky cache logic) from {config.MODEL_PATH}...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model and Tokenizer loaded successfully.")

    data_dir = getattr(config, "DATA_DIR", "1LongBenchData")
    print(f"Loading datasets from {data_dir}...")
    datasets = data_loader.load_datasets(data_dir)

    all_results = {}

    for task_name, dataset in datasets.items():
        print(f"\nEvaluating dataset: {task_name}")
        for seed in config.SEEDS:
            res = engine.evaluate_dataset(
                name=task_name,
                dataset=dataset,
                seed=seed,
                model=model,
                tokenizer=tokenizer,
                device="cuda"
            )

            if task_name not in all_results:
                all_results[task_name] = []
            all_results[task_name].append(res)

            print(f"Completed {task_name} (Seed {seed}): Evaluated {res['sample_size']} instances.")

    import numpy as np
    output_file = "long_bench_pure_vanilla_metrics.npz"
    np.savez_compressed(output_file, data=np.array([json.dumps(all_results)]))

    print(f"\nAll evaluations complete. Results saved to {output_file}")


if __name__ == "__main__":
    main()
