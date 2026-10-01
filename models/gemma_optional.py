"""Optional Gemma 3 4B wrapper. Not used by the core pipeline."""
from __future__ import annotations

from config import Settings
from models.model_loader import load_causal_lm


class OptionalGemma:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.tokenizer = None
        self.model = None

    def load(self):
        if not self.settings.load_optional_gemma:
            raise RuntimeError("Gemma is disabled. Set LOAD_OPTIONAL_GEMMA=true to enable it.")
        self.tokenizer, self.model = load_causal_lm(
            self.settings.gemma_model_id,
            self.settings.torch_device,
            self.settings.hf_cache_dir,
            quantize=True,
        )
        return self

    def generate(self, prompt: str, max_new_tokens: int = 300) -> str:
        if self.model is None:
            self.load()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
