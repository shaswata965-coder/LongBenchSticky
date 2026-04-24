
from transformers.configuration_utils import PretrainedConfig

class LlamaConfig(PretrainedConfig):
    model_type = "llama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=128256,
        hidden_size=2048,          # Fixed: 4096 -> 2048
        intermediate_size=8192,     # Fixed: 14336 -> 8192
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=128000,
        eos_token_id=128001,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=500000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        p_ratio=20,
        r_ratio=50,
        start_idx=0,
        **kwargs,
    ):
        # Pass ALL relevant parameters to parent, including tie_word_embeddings
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        
        # Set each attribute ONCE
        self.p_ratio = p_ratio
        self.r_ratio = r_ratio
        self.start_idx = start_idx
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        self._rope_scaling_validation()

    def _rope_scaling_validation(self):
        """
        Updated validation for Llama 3.2 compatibility.
        """
        if self.rope_scaling is None:
            return
    
        if not isinstance(self.rope_scaling, dict):
            raise ValueError("`rope_scaling` must be a dictionary.")
    
        # Llama 3.1/3.2 models often use 'rope_type' instead of 'type'
        rope_type = self.rope_scaling.get("type") or self.rope_scaling.get("rope_type")
        rope_factor = self.rope_scaling.get("factor")
    
        if rope_type is None:
            raise ValueError("`rope_scaling` must contain either 'type' or 'rope_type'.")
    
        # Expand allowed types to include 'llama3'
        allowed_types = ["linear", "dynamic", "llama3"]
        if rope_type not in allowed_types:
            raise ValueError(f"`rope_scaling` type must be one of {allowed_types}, got {rope_type}")
    
        if rope_factor is not None:
            if not isinstance(rope_factor, float) or rope_factor < 1.0:
                raise ValueError("`rope_scaling` factor must be a float >= 1.")