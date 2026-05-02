"""Sticky-KV fields on top of HuggingFace Qwen2Config (Qwen 2.5, etc.)."""

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


class StickyQwen2Config(Qwen2Config):
    """Qwen2 config with sticky eviction hyperparameters."""

    model_type = "qwen2"

    def __init__(self, **kwargs):
        p_ratio = kwargs.pop("p_ratio", 50)
        r_ratio = kwargs.pop("r_ratio", 50)
        start_idx = kwargs.pop("start_idx", 0)
        # Must survive HF (de)serialization so model init can honor it.
        use_fast_attention = kwargs.pop("use_fast_attention", True)
        super().__init__(**kwargs)
        self.p_ratio = p_ratio
        self.r_ratio = r_ratio
        self.start_idx = start_idx
        self.use_fast_attention = use_fast_attention
