import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM




class STICKYQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        use_fast_attention = getattr(config, "use_fast_attention", True)
        if use_fast_attention:
            from sticky_qwen2_attention_fast_attention import STICKYQwen2Attention
            print("DEBUG: Using FAST STICKYQwen2Attention backend.")
        else:
            from sticky_qwen2_attention import STICKYQwen2Attention
            print("DEBUG: Using CUMMULATIVE STICKYQwen2Attention backend.")
            
        print(f"DEBUG: Initializing STICKYQwen2ForCausalLM with {len(self.model.layers)} layers")
        for layer_idx in range(len(self.model.layers)):
            self.model.layers[layer_idx].self_attn = STICKYQwen2Attention(config, layer_idx)
        print("DEBUG: All Qwen2 attention layers replaced.")

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """Match sticky physical KV length to logical positions (same idea as Llama sticky)."""
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        past_length = 0
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                past_length = past_key_values.get_seq_length()
            elif isinstance(past_key_values, tuple) and len(past_key_values) > 0 and past_key_values[0] is not None:
                past_length = past_key_values[0][0].shape[2]

        if past_length > 0:
            model_inputs["input_ids"] = input_ids[:, -1:]
            position_ids = kwargs.get("position_ids", None)
            if position_ids is None:
                if attention_mask is not None:
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)
                    model_inputs["position_ids"] = position_ids[:, -1:]
                else:
                    true_seq_length = input_ids.shape[1] + past_length
                    model_inputs["position_ids"] = torch.tensor(
                        [[true_seq_length - 1]], dtype=torch.long, device=input_ids.device
                    )
            else:
                model_inputs["position_ids"] = position_ids[:, -1:]
        return model_inputs
