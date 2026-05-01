import torch
from transformers import AutoTokenizer
from configuration_sticky_llama import LlamaConfig
from sticky_llama_model import STICKYLlamaForCausalLM
import sticky_config

def main():
    # 1. Path to your local model weights/directory
    model_path = sticky_config.MODEL_PATH

    # 2. Define your custom configuration
    config = LlamaConfig.from_pretrained(model_path)
    
    # Configuration Option A: Use a percentage of the cache (e.g., 20% local cache ratio)
    # config.p_ratio = 20  
    
    # Configuration Option B: Use a fixed number of recent tokens (e.g., 256 tokens)
    # FIX (M7): Set LOCAL_NUM_TOKENS on the sticky_config MODULE, not on config.
    # STICKYKVCache_LayerWise.__init__ reads from `from sticky_config import LOCAL_NUM_TOKENS`,
    # so config.local_num_tokens has no effect when that import succeeds.
    sticky_config.LOCAL_NUM_TOKENS = 256
    
    config.r_ratio = getattr(sticky_config, "R_RATIO", 50)
    config.start_idx = getattr(sticky_config, "S_IDX", 0)

    print(f"Loading model with Config: LOCAL_NUM_TOKENS={sticky_config.LOCAL_NUM_TOKENS}, r_ratio={config.r_ratio}")

    # 3. Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Ensure tokenizer has proper padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer chat template available: {tokenizer.chat_template is not None}")
    
    # Fix the rope_scaling "Version Gap" for Llama 3.2 compatibility
    # 4.36.0 expects 'type' or nothing. Llama 3.2 provides 'rope_type'.
    if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        # Option A: Force it to a format v4.36 knows
        if "rope_type" in config.rope_scaling and "type" not in config.rope_scaling:
            config.rope_scaling["type"] = config.rope_scaling["rope_type"]
        # Option B: Preserve it! Our custom attention layer now handles it.
        # config.rope_scaling = None 
    
    # Ensure the Llama 3 theta is preserved
    config.rope_theta = getattr(config, "rope_theta", 500000.0)
    # 4. Initialize the custom model with the local weights
    model = STICKYLlamaForCausalLM.from_pretrained(
        model_path, 
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    
    print(f"Model loaded on device: {model.device}")
    print(f"Model dtype: {model.dtype}")
    print(f"Rope Scaling Config: {model.config.rope_scaling}")
    # 5. Create properly formatted prompt using chat template
    messages = [
        {"role": "user", "content": "The history of the Valkyria Chronicles series is deeply intertwined with its unique strategic RPG combat system known as BLiTZ (Battle of Live Tactical Zones). Developed by Sega, the franchise first debuted on the PlayStation 3 in 2008 and quickly garnered a passionate fanbase due to its gorgeous CANVAS graphics engine, which mimics the appearance of a watercolor painting in motion. The narrative frequently centers around the small, neutral nation of Gallia as it struggles to maintain its independence against the overwhelming military might of the East Europan Imperial Alliance.In Valkyria Chronicles 3, the story shifts focus to a highly secretive penal military unit known as Squad 422, or 'The Nameless.'These individuals have been stripped of their names and identities, referred to only by numbers, and are tasked with executing highly dangerous black-ops missions that the regular Gallian army cannot legally undertake. The squad is completely expendable, composed of criminals, insubordinate soldiers, and those falsely accused of treason. Kurt Irving, an incredibly gifted military tactician falsely disgraced by a conspiracy, is assigned as their new commander. He must not only lead this ragtag group of outcasts to survive seemingly impossible suicide missions but also uncover the truth behind his own downfall to clear his name. The squad features a diverse cast of tragic characters, such as Riela Marcellis, a young woman ostracized for her mysterious resilience to death which stems from her latent Valkyria blood, and Imca, a fiercely independent Darcsen warrior driven entirely by a singular desire for revenge against the Imperial soldier who destroyed her village. Throughout their grueling campaign, The Nameless must face off against Calamity Raven, a formidable Imperial black-ops unit consisting entirely of Darcsens who are fighting for a promised autonomous homeland. As the war escalates, Kurt and his squad find themselves entangled in deeply complicated political machinations, constantly manipulated by their own corrupt Gallian commanders while desperately fighting to regain their true identities and prove their irreplaceable worth on the battlefield.Please write a comprehensive, detailed 200-word continuation expanding on the following text. Do not stop early."}
    ]
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    print(f"\n=== Formatted Prompt ===")
    print(prompt)
    print("=" * 50)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"Input shape: {inputs['input_ids'].shape}")
    
    # 6. Generate with proper parameters
    print("\nGenerating output...")
    # Llama 3 instruct models often use <|eot_id|> as terminator
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    output = model.generate(
        **inputs, 
        **sticky_config.GENERATION_CONFIG,
        repetition_penalty=1.1,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True
    )

    input_len = inputs["input_ids"].shape[1]
    clean_output = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
    
    print("\n=== Clean Output ===")
    print(clean_output)
    

if __name__ == "__main__":
    main()