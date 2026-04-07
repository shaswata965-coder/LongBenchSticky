from transformers.models.llama.modeling_llama import LlamaForCausalLM
from sticky_llama_attention_fast_attention import STICKYLlamaAttention

import copy # ensure copy is imported if not already

class STICKYLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        # The parent LlamaForCausalLM initializes LlamaModel, which initializes LlamaAttention.
        # Standard LlamaAttention validates 'rope_scaling'. If it sees 'llama3' (unknown to older transformers), it crashes.
        # We must strip it for the parent init, then put the correct config back for OUR custom layers.
        
        # 1. Create a "safe" config for the parent class
        safe_config = copy.deepcopy(config)
        if hasattr(safe_config, "rope_scaling"):
            safe_config.rope_scaling = None
            
        # 2. Initialize parent with safe config
        super().__init__(safe_config)
        
        # 3. Restore the correct config on the model instance (so model.config is correct)
        self.config = config
        
        print(f"DEBUG: Initializing STICKYLlamaForCausalLM with {len(self.model.layers)} layers")
        
        for layer_idx in range(len(self.model.layers)):
            # Explicitly overwrite the module using the ORIGINAL (unsafe) config
            self.model.layers[layer_idx].self_attn = STICKYLlamaAttention(config, layer_idx)
        
        print("DEBUG: All attention layers replaced.")

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs
        )
        if past_key_values is not None:
            # Override incorrect slicing done by super() due to evicted cache size
            model_inputs["input_ids"] = input_ids[:, -1:]
            
            # Position IDs generation needs to account for the total generated length
            # because the KV cache has been artificially shortened by eviction.
            # `input_ids.shape[1]` perfectly tracks the true global sequence length
            # during transformers `.generate()` loops.
            position_ids = kwargs.get("position_ids", None)
            if position_ids is None:
                if attention_mask is not None:
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)
                    model_inputs["position_ids"] = position_ids[:, -1:]
                else:
                    true_seq_length = input_ids.shape[1]
                    model_inputs["position_ids"] = torch.tensor([[true_seq_length - 1]], dtype=torch.long, device=input_ids.device)
            else:
                model_inputs["position_ids"] = position_ids[:, -1:]
        
        return model_inputs