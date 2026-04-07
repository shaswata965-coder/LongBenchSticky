import sys
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directory to path to import engine, data_loader, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine
import data_loader
import sticky_config as config

def main():
    print(f"Loading Pure Vanilla LLaMA (No sticky cache logic) from {config.MODEL_PATH}...")
    
    # Load base HF LLaMA 3.2 1B
    try:
        from transformers.models.llama.configuration_llama import LlamaConfig as HFLlamaConfig
        with open(os.path.join(config.MODEL_PATH, "config.json"), "r") as f:
            v_config_dict = json.load(f)
        if "rope_scaling" in v_config_dict:
            del v_config_dict["rope_scaling"]
        v_config = HFLlamaConfig(**v_config_dict)

        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_PATH, 
            config=v_config,
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

    # Load LongBench datasets
    data_dir = "1LongBenchData"
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
            
            # Aggregate or store the result
            if task_name not in all_results:
                all_results[task_name] = []
            all_results[task_name].append(res)
            
            # Print intermediate sample size completion
            print(f"Completed {task_name} (Seed {seed}): Evaluated {res['sample_size']} instances.")

    # Save outputs as NPZ
    import numpy as np
    output_file = "long_bench_pure_vanilla_metrics.npz"
    np.savez_compressed(output_file, data=np.array([json.dumps(all_results)]))
    
    print(f"\nAll evaluations complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
