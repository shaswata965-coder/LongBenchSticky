import sys
import os
import json
import time
import argparse
import torch
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine
import data_loader
import sticky_config as config
from sticky_llama_model import STICKYLlamaForCausalLM
from configuration_sticky_llama import LlamaConfig


def main():
    print(f"Loading StickyLlama (Custom Cumulative Logic) from {config.MODEL_PATH}...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated dataset names (e.g. 2wikimqa,hotpotqa)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda"

    data_dir = "/home/ee/phd/eez228470/kv_cache/defensive_kv_new/DefensiveKV/Final_LongBench_Dataset"

    if args.tasks:
        test_tasks = [t.strip() for t in args.tasks.split(",")]
    else:
        test_tasks = sorted(f.replace(".jsonl", "")
                            for f in os.listdir(data_dir) if f.endswith(".jsonl"))

    print(f"[GPU {args.gpu}] Loading model from {config.MODEL_PATH}...")

    try:
        model_config = LlamaConfig.from_pretrained(config.MODEL_PATH)

        if hasattr(model_config, "rope_scaling") and model_config.rope_scaling is not None:
            if "rope_type" in model_config.rope_scaling and "type" not in model_config.rope_scaling:
                model_config.rope_scaling["type"] = model_config.rope_scaling["rope_type"]

        model_config.rope_theta = getattr(model_config, "rope_theta", 500000.0)
        model_config.r_ratio = getattr(config, "R_RATIO", 50)

        if hasattr(config, "P_RATIO"):
            model_config.p_ratio = config.P_RATIO
        elif hasattr(config, "LOCAL_NUM_TOKENS"):
            model_config.local_num_tokens = config.LOCAL_NUM_TOKENS

        model_config.start_idx = getattr(config, "S_IDX", 0)

        model = STICKYLlamaForCausalLM.from_pretrained(
            config.MODEL_PATH,
            config=model_config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    except Exception as e:
        print(f"[GPU {args.gpu}] Error loading model: {e}")
        return

    print(f"[GPU {args.gpu}] Model loaded. Tasks: {test_tasks}")

    datasets = {}
    for t in test_tasks:
        path = os.path.join(data_dir, f"{t}.jsonl")
        if not os.path.exists(path):
            print(f"[GPU {args.gpu}] WARNING: {t}.jsonl not found, skipping")
            continue
        ds = data_loader.load_jsonl(path)
        datasets[t] = ds
        max_tok = data_loader.max_new_tokens.get(t, config.GENERATION_CONFIG["max_new_tokens"])
        print(f"  Loaded {t} ({len(ds)} samples, max_new_tokens={max_tok})")

    all_results = {}
    total_start = time.time()

    for task_name, dataset in datasets.items():
        print(f"\n[GPU {args.gpu}] Evaluating: {task_name}")
        for seed in config.SEEDS:
            task_start = time.time()
            res = engine.evaluate_dataset(
                name=task_name,
                dataset=dataset,
                seed=seed,
                model=model,
                tokenizer=tokenizer,
                device=device
            )
            task_elapsed = time.time() - task_start

            if task_name not in all_results:
                all_results[task_name] = []
            all_results[task_name].append(res)

            n = res['sample_size']
            per_sample = task_elapsed / n if n > 0 else 0
            print(f"[GPU {args.gpu}] Done {task_name} (Seed {seed}): "
                  f"{n} samples in {task_elapsed:.1f}s ({per_sample:.2f}s/sample)")

    total_elapsed = time.time() - total_start
    print(f"\n[GPU {args.gpu}] Total: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    import numpy as np
    output_file = f"results_gpu{args.gpu}.npz"
    np.savez_compressed(output_file, data=np.array([json.dumps(all_results)]))
    print(f"[GPU {args.gpu}] Saved to {output_file}")


if __name__ == "__main__":
    main()
