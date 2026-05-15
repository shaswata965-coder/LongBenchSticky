import torch
import copy
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from sticky_llama_attention import STICKYLlamaAttention

class STICKYLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, **kwargs):
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
        """Prepare inputs for autoregressive generation.
        
        NOTE: The position_ids computed here are overridden by
        STICKYLlamaAttention.forward() which uses global_token_counter for RoPE.
        This override is necessary because the KV cache is compressed by eviction,
        so the framework's position tracking is incorrect.
        
        WARNING: batch_size > 1 is structurally unsupported because
        global_token_counter is per-layer, not per-batch-item.
        """
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs
        )
        if past_key_values is not None:
            prep_dbg_count = getattr(self, "_dbg_prepare_count", 0)

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

            if prep_dbg_count < 10:
                cache_pos = model_inputs.get("cache_position", None)
                pos_ids = model_inputs.get("position_ids", None)
                cache_pos_dbg = cache_pos.detach().flatten().tolist() if torch.is_tensor(cache_pos) else cache_pos
                pos_ids_dbg = pos_ids.detach().flatten().tolist() if torch.is_tensor(pos_ids) else pos_ids
                print(
                    f"[GEN-PREP original step={prep_dbg_count}] "
                    f"input_len={input_ids.shape[1]} sliced_len={model_inputs['input_ids'].shape[1]} "
                    f"cache_position={cache_pos_dbg} position_ids={pos_ids_dbg} "
                    f"pkv_type={type(past_key_values).__name__}",
                    flush=True,
                )
            self._dbg_prepare_count = prep_dbg_count + 1
        
        return model_inputs
