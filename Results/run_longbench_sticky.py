import sys
import os
import json
import torch
from transformers import AutoTokenizer

# Add parent directory to path to import engine, data_loader, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine
import data_loader
import sticky_config as config
from sticky_llama_model import STICKYLlamaForCausalLM
from configuration_sticky_llama import LlamaConfig

def main():
    print(f"Loading StickyLlama (Custom Cumulative Logic) from {config.MODEL_PATH}...")
    
    # Load Sticky LLaMA
    try:
        model_config = LlamaConfig.from_pretrained(config.MODEL_PATH)
        
        # Resolve RoPE configurations
        if hasattr(model_config, "rope_scaling") and model_config.rope_scaling is not None:
            if "rope_type" in model_config.rope_scaling and "type" not in model_config.rope_scaling:
                model_config.rope_scaling["type"] = model_config.rope_scaling["rope_type"]
        
        model_config.rope_theta = getattr(model_config, "rope_theta", 500000.0)
        
        # Inject custom sticky config variables
        model_config.r_ratio = getattr(config, "R_RATIO", 50)
        
        # Either P_RATIO or LOCAL_NUM_TOKENS will be present
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
        print(f"Error loading model: {e}")
        return

    print("Model and Tokenizer loaded successfully.")

    # Load LongBench datasets
    data_dir = "1LongBenchData"
    print(f"Loading datasets from {data_dir}...")
    datasets = data_loader.load_datasets(data_dir)

    all_results = {}
    
    for task_name, dataset in datasets.items():
        print(f"\nEvaluating dataset: {task_name}")
        
        # Fetch dataset specific max token limit
        task_max_tokens = data_loader.max_new_tokens.get(task_name, 128)
        
        for seed in config.SEEDS:
            # We don't need to manually clear the cache here because engine.py
            # has been updated with automatic scrubbing inside pre-generation cleanup.
            res = engine.evaluate_dataset(
                name=task_name,
                dataset=dataset,
                seed=seed,
                model=model,
                tokenizer=tokenizer,
                device="cuda",
                max_tokens=task_max_tokens
            )
            
            # Aggregate or store the result
            if task_name not in all_results:
                all_results[task_name] = []
            all_results[task_name].append(res)
            
            # Print intermediate sample size completion
            print(f"Completed {task_name} (Seed {seed}): Evaluated {res['sample_size']} instances.")

    # Save outputs as NPZ
    import numpy as np
    output_file = "long_bench_sticky_metrics.npz"
    np.savez_compressed(output_file, data=np.array([json.dumps(all_results)]))
    
    print(f"\nAll evaluations complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
