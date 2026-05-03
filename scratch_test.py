import torch
import math

bsz = 1
q_len = 2
num_heads = 28
num_key_value_heads = 4
num_key_value_groups = 7
head_dim = 128
kv_seq_len = 10

query_states = torch.randn(bsz, num_heads, q_len, head_dim)
key_states = torch.randn(bsz, num_key_value_heads, kv_seq_len, head_dim)

# My method
q_grouped = query_states.reshape(bsz, num_key_value_heads, num_key_value_groups, q_len, head_dim)
main_logits = torch.matmul(q_grouped, key_states.transpose(2, 3).unsqueeze(2)) / math.sqrt(head_dim)
main_logits_my = main_logits.reshape(bsz, num_heads, q_len, -1)

# HF method
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

key_states_hf = repeat_kv(key_states, num_key_value_groups)
main_logits_hf = torch.matmul(query_states, key_states_hf.transpose(2, 3)) / math.sqrt(head_dim)

print("Max diff:", (main_logits_my - main_logits_hf).abs().max().item())
